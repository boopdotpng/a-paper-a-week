from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import math
import itertools
import os
import re

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")
if device.type == "cuda":
  torch.backends.cudnn.benchmark = True
  torch.backends.cuda.matmul.fp32_precision = "tf32"
  torch.backends.cudnn.conv.fp32_precision = "tf32"
  torch.set_float32_matmul_precision("high")

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
  train_data,
  batch_size=512,
  shuffle=True,
  num_workers=4,
  pin_memory=device.type == "cuda",
  persistent_workers=True,
  prefetch_factor=4,
  drop_last=True,
)

def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
  args = t[:, None] * freqs[None]
  return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
    super().__init__()
    self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.t_proj = nn.Linear(t_dim, out_ch)
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

  def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
    h = self.conv1(F.silu(self.norm1(x)))
    h = h + self.t_proj(t_emb)[:, :, None, None]
    h = self.conv2(F.silu(self.norm2(h)))
    return h + self.skip(x)


class Downsample(nn.Module):
  def __init__(self, ch: int):
    super().__init__()
    self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class Upsample(nn.Module):
  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.tconv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.tconv(x)


class Model(nn.Module):
  def __init__(self, base_ch: int = 64, t_dim: int = 256):
    super().__init__()
    self.t_dim = t_dim
    self.t_mlp = nn.Sequential(
      nn.Linear(t_dim, t_dim * 4),
      nn.SiLU(),
      nn.Linear(t_dim * 4, t_dim),
      nn.SiLU(),
    )

    # encoder
    self.e1 = ResBlock(3, base_ch, t_dim)
    self.d1 = Downsample(base_ch)
    self.e2 = ResBlock(base_ch, base_ch * 2, t_dim)
    self.d2 = Downsample(base_ch * 2)
    self.e3 = ResBlock(base_ch * 2, base_ch * 4, t_dim)
    self.d3 = Downsample(base_ch * 4)
    self.e4 = ResBlock(base_ch * 4, base_ch * 8, t_dim)
    self.d4 = Downsample(base_ch * 8)

    # bottleneck
    self.mid1 = ResBlock(base_ch * 8, base_ch * 16, t_dim)
    self.mid2 = ResBlock(base_ch * 16, base_ch * 16, t_dim)

    # decoder
    self.u4 = Upsample(base_ch * 16, base_ch * 8)
    self.de4 = ResBlock(base_ch * 16, base_ch * 8, t_dim)
    self.u3 = Upsample(base_ch * 8, base_ch * 4)
    self.de3 = ResBlock(base_ch * 8, base_ch * 4, t_dim)
    self.u2 = Upsample(base_ch * 4, base_ch * 2)
    self.de2 = ResBlock(base_ch * 4, base_ch * 2, t_dim)
    self.u1 = Upsample(base_ch * 2, base_ch)
    self.de1 = ResBlock(base_ch * 2, base_ch, t_dim)

    self.out = nn.Conv2d(base_ch, 3, 1)

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))

    s1 = self.e1(x, t_emb)
    x = self.d1(s1)
    s2 = self.e2(x, t_emb)
    x = self.d2(s2)
    s3 = self.e3(x, t_emb)
    x = self.d3(s3)
    s4 = self.e4(x, t_emb)
    x = self.d4(s4)

    x = self.mid1(x, t_emb)
    x = self.mid2(x, t_emb)

    x = self.u4(x)
    x = torch.cat([x, s4], dim=1)
    x = self.de4(x, t_emb)
    x = self.u3(x)
    x = torch.cat([x, s3], dim=1)
    x = self.de3(x, t_emb)
    x = self.u2(x)
    x = torch.cat([x, s2], dim=1)
    x = self.de2(x, t_emb)
    x = self.u1(x)
    x = torch.cat([x, s1], dim=1)
    x = self.de1(x, t_emb)

    return self.out(x)


def _image_from_sample(sample: torch.Tensor, scale: int = 1) -> Image.Image:
  sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
  sample = sample.permute(1, 2, 0).contiguous()
  img = Image.fromarray(sample.cpu().numpy())
  if scale > 1:
    img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
  return img


@torch.no_grad()
def sample_heun(
  model: nn.Module,
  n_steps: int = 200,
  bs: int = 16,
  eps: float = 1e-3,
  use_amp: bool = False,
  gif_every: int | None = None,
  scale: int = 1,
) -> tuple[torch.Tensor, list[Image.Image] | None]:
  model.eval()
  amp_dtype = torch.bfloat16 if use_amp else torch.float32
  amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
  t0, t1 = eps, 1.0
  dt = (t1 - t0) / n_steps
  t_steps = torch.linspace(t0, t1, n_steps, device=device, dtype=torch.float32)

  x = torch.randn(bs, 3, 32, 32, device=device, dtype=torch.float32)
  frames: list[Image.Image] | None = [] if gif_every is not None else None
  if frames is not None:
    frames.append(_image_from_sample(x[0], scale=scale))

  with amp_ctx:
    for i in range(n_steps - 1):
      ti = t_steps[i].expand(bs)
      tip1 = t_steps[i + 1].expand(bs)

      v = model(x, ti)
      x_euler = x + dt * v
      v_next = model(x_euler, tip1)
      x = x + dt * 0.5 * (v + v_next)
      if frames is not None and (i + 1) % gif_every == 0:
        frames.append(_image_from_sample(x[0], scale=scale))

    # final euler step
    ti = t_steps[-1].expand(bs)
    v = model(x, ti)
    x = x + dt * v

  return x, frames


def main():
  import sys
  CHECKPOINT_DIR.mkdir(exist_ok=True)
  OUTPUT_DIR.mkdir(exist_ok=True)
  model = Model().to(device, memory_format=torch.channels_last)
  optim = torch.optim.Adam(model.parameters(), lr=2e-4)
  use_amp = device.type == "cuda"
  amp_dtype = torch.bfloat16 if use_amp else torch.float32
  scaler = torch.amp.GradScaler(enabled=use_amp)
  amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
  gif_every = 10 if "--gif" in sys.argv else None
  gif_duration_ms = 120
  gif_hold_ms = 1200
  gif_scale = 4
  sample_only = "--sample" in sys.argv

  def _latest_ckpt_path() -> str | None:
    ckpt_paths = [p for p in CHECKPOINT_DIR.iterdir() if re.match(r"ckpt_\d+\.pt$", p.name)]
    return str(max(ckpt_paths, key=lambda p: int(p.name.split("_")[1].split(".")[0]))) if ckpt_paths else None

  def _load_checkpoint(path: str) -> int:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state.keys()):
      state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state)
    if "optim" in ckpt:
      optim.load_state_dict(ckpt["optim"])
    return ckpt.get("step", int(path.split("_")[1].split(".")[0]))

  resume_path = _latest_ckpt_path()
  start_step = 0
  if resume_path:
    start_step = _load_checkpoint(resume_path)
    print(f"resumed from {resume_path} @ step {start_step}")

  if use_amp and not sample_only:
    model = torch.compile(model)

  if not sample_only:
    data_iter = itertools.cycle(train_loader)

    for step in range(start_step + 1, 20001):
      model.train()

      imgs, _ = next(data_iter)

      imgs = imgs.to(device, dtype=torch.float32, non_blocking=True, memory_format=torch.channels_last)
      noises = torch.randn_like(imgs)
      t = torch.rand(imgs.shape[0], device=device, dtype=torch.float32)

      with amp_ctx:
        t_img = t[:, None, None, None]
        xt = (1 - t_img) * noises + t_img * imgs
        v_target = imgs - noises
        v_pred = model(xt, t)
        loss = F.mse_loss(v_pred, v_target)

      optim.zero_grad(set_to_none=True)
      scaler.scale(loss).backward()
      scaler.step(optim)
      scaler.update()

      if step % 100 == 0:
        print(f"loss: {loss.item():.2f}, step {step}")
      if step % 1000 == 0:
        ckpt_path = CHECKPOINT_DIR / f"ckpt_{step}.pt"
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step}, ckpt_path)

  if gif_every is not None or sample_only:
    resume_path = _latest_ckpt_path()
    if resume_path is None:
      print("no checkpoint found for sampling")
      return
    _load_checkpoint(resume_path)
    print(f"loaded {resume_path} for sampling")
    if use_amp and not hasattr(model, "_orig_mod"):
      model = torch.compile(model)

  # generate samples
  samples, frames = sample_heun(model, n_steps=200, bs=1, use_amp=use_amp, gif_every=gif_every, scale=gif_scale)
  img = _image_from_sample(samples[0], scale=gif_scale)
  output_path = OUTPUT_DIR / "samples_torch.png"
  img.save(output_path)
  print(f"saved {output_path}")
  if frames is not None:
    final_hold_frames = max(1, gif_hold_ms // gif_duration_ms)
    frames.extend([frames[-1]] * final_hold_frames)
    gif_path = OUTPUT_DIR / "samples_torch.gif"
    frames[0].save(
      gif_path,
      save_all=True,
      append_images=frames[1:],
      duration=gif_duration_ms,
      loop=0,
      disposal=2,
    )
    print(f"saved {gif_path}")


if __name__ == "__main__":
  main()
