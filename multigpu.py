from pathlib import Path
from tinygrad import Tensor, TinyJit, dtypes, Device
from PIL import Image
import numpy as np
from tinygrad.helpers import GlobalCounters, Context
from tinygrad.nn.datasets import cifar
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load
from tinygrad.nn.optim import Adam
import tinygrad.nn as nn
import math

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

# Multi-GPU setup
GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(2))  # Use 2 GPUs
print(f"Using GPUs: {GPUS}")

train_x, train_y, test_x, test_y = cifar()
train_x = (train_x / 127.5 - 1.0).half()
print(train_x.shape, train_x.dtype)

# Shard training data across GPUs on batch axis
train_x.shard_(GPUS, axis=0)
train_y.shard_(GPUS, axis=0)
print(f"Training data sharded across {len(GPUS)} GPUs")

Tensor.default_type = dtypes.half

_timestep_freqs_cache: dict[int, Tensor] = {}

def timestep_embedding(t: Tensor, dim: int = 256) -> Tensor:
  half = dim // 2
  if dim not in _timestep_freqs_cache:
    _timestep_freqs_cache[dim] = Tensor.exp(-math.log(10000) * Tensor.arange(0, half) / half).realize()
  freqs = _timestep_freqs_cache[dim]
  args = t.reshape(-1, 1) * freqs.reshape(1, -1)                         # (B, half)
  return args.sin().cat(args.cos(), dim=1)                               # (B, dim)


class ResBlock:
  def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
    self.norm1 = nn.GroupNorm(groups, in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.norm2 = nn.GroupNorm(groups, out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    self.t_proj = nn.Linear(t_dim, out_ch)
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

  def __call__(self, x: Tensor, t_emb: Tensor):
    h = self.conv1(Tensor.silu(self.norm1(x)))
    h = h + self.t_proj(t_emb).reshape(x.shape[0], -1, 1, 1)
    h = self.conv2(Tensor.silu(self.norm2(h)))
    res = self.skip(x) if self.skip is not None else x
    return h + res


class Downsample:
  def __init__(self, ch: int):
    self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
  def __call__(self, x: Tensor) -> Tensor:
    return self.conv(x)

class Upsample:
  def __init__(self, in_ch:int, out_ch: int):
    self.tconv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
  def __call__(self, x: Tensor) -> Tensor:
    return self.tconv(x)

class Model:
  def __init__(self, base_ch:int = 64, t_dim:int = 256):
    # time dimension embedding
    self.t_dim= t_dim
    self.t_mlp = [
        nn.Linear(self.t_dim, self.t_dim*4),
        Tensor.silu,
        nn.Linear(self.t_dim*4, self.t_dim)
    ]
    
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
    self.de4 = ResBlock(base_ch * 16, base_ch * 8, t_dim)  # concat doubles ch

    self.u3 = Upsample(base_ch * 8, base_ch * 4)
    self.de3 = ResBlock(base_ch * 8, base_ch * 4, t_dim)

    self.u2 = Upsample(base_ch * 4, base_ch * 2)
    self.de2 = ResBlock(base_ch * 4, base_ch * 2, t_dim)

    self.u1 = Upsample(base_ch * 2, base_ch)
    self.de1 = ResBlock(base_ch * 2, base_ch, t_dim)

    self.out = nn.Conv2d(base_ch, 3, 1)

  def __call__(self, x: Tensor, t: Tensor) -> Tensor:
    t_emb = timestep_embedding(t, self.t_dim)
    t_emb = t_emb.sequential(self.t_mlp)
    t_emb = Tensor.silu(t_emb)

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
    x = x.cat(s4, dim=1)
    x = self.de4(x, t_emb)

    x = self.u3(x) 
    x = x.cat(s3, dim=1)
    x = self.de3(x, t_emb)

    x = self.u2(x)
    x = x.cat(s2, dim=1)
    x = self.de2(x, t_emb)

    x = self.u1(x)
    x = x.cat(s1, dim=1)
    x = self.de1(x, t_emb)

    return self.out(x)

model = Model()
ema_model = Model()

# Replicate model weights across all GPUs (axis=None for replication)
for x in get_state_dict(model).values():
  x.to_(GPUS)
for x in get_state_dict(ema_model).values():
  x.to_(GPUS)
print(f"Model replicated across {len(GPUS)} GPUs")

# initialize EMA weights to match model
for p, p_ema in zip(get_parameters(model), get_parameters(ema_model)):
  p_ema.assign(p).realize()

# Batch size must divide evenly across GPUs for multi-GPU training
batch = 256  # 128 per GPU with 2 GPUs
assert batch % len(GPUS) == 0, f"Batch size {batch} must divide evenly across {len(GPUS)} GPUs"

optim = Adam(get_parameters(model), lr=1e-4)
ema_decay = 0.9999

@TinyJit
@Tensor.train()
def train():
  # generate new intial states (32x32 rgb gaussian noise image)
  noises = Tensor.randn(batch, 3, 32, 32)

  # pick random images from cifar training set 
  random_idx = Tensor.randint(batch, high=train_x.shape[0])
  imgs = train_x[random_idx] # batch, 3, 32, 32

  # interpolate between random noise and real image 
  t = Tensor.rand(batch)
  t_img = t.reshape(batch, 1, 1, 1)
  xt = (1-t_img) * noises + t_img * imgs 
  v_target = imgs - noises

  v_pred = model(xt, t)

  loss = (v_pred - v_target).square().mean()

  optim.zero_grad()
  loss.backward()  # Gradients auto-synced via allreduce across GPUs
  optim.step()
  
  return loss

def update_ema():
  for p, p_ema in zip(get_parameters(model), get_parameters(ema_model)):
    p_ema.assign(ema_decay * p_ema + (1 - ema_decay) * p).realize()

# Module level JITs - persist across calls
_heun_jit = None
_euler_jit = None

def sample_heun(model: Model, n_steps: int = 200, bs: int = 16, eps: float = 1e-3) -> Tensor:
  global _heun_jit, _euler_jit

  t0, t1 = eps, 1.0
  dt = (t1 - t0) / n_steps

  Tensor.training = False

  # Create JITs once
  if _heun_jit is None:
    @TinyJit
    def heun_step(x: Tensor, ti: Tensor, tip1: Tensor, dt_t: Tensor) -> Tensor:
      v = model(x, ti)
      x_euler = x + dt_t * v
      v_next = model(x_euler, tip1)
      return (x + dt_t * 0.5 * (v + v_next)).realize()
    _heun_jit = heun_step

  if _euler_jit is None:
    @TinyJit
    def euler_step(x: Tensor, ti: Tensor, dt_t: Tensor) -> Tensor:
      v = model(x, ti)
      return (x + dt_t * v).realize()
    _euler_jit = euler_step

  x = Tensor.randn(bs, 3, 32, 32).realize()

  # Pre-allocate reusable buffers (JIT inputs can't be CONSTs)
  dt_t = Tensor([dt]).realize()
  ti_buf = Tensor.zeros(bs).contiguous().realize()
  tip1_buf = Tensor.zeros(bs).contiguous().realize()

  for i in range(n_steps - 1):
    ti_buf.assign(Tensor.full((bs,), t0 + i * dt)).realize()
    tip1_buf.assign(Tensor.full((bs,), t0 + (i + 1) * dt)).realize()
    x = _heun_jit(x, ti_buf, tip1_buf, dt_t)

  # final euler step
  ti_buf.assign(Tensor.full((bs,), t0 + (n_steps - 1) * dt)).realize()
  x = _euler_jit(x, ti_buf, dt_t)

  return x


def main():
  import sys
  skip_train = "--sample" in sys.argv
  CHECKPOINT_DIR.mkdir(exist_ok=True)
  OUTPUT_DIR.mkdir(exist_ok=True)

  if skip_train:
    ckpt_path = CHECKPOINT_DIR / "ema_model.safetensors"
    load_state_dict(ema_model, safe_load(str(ckpt_path)))
    print(f"loaded {ckpt_path}")
  else:
    for i in range(1, 20001):
      loss = train()
      update_ema()
      if i % 100 == 0:
        print(f"loss: {loss.item():.2f}, step {i}")
      if i % 5000 == 0:
        ckpt_path = CHECKPOINT_DIR / "ema_model.safetensors"
        safe_save(get_state_dict(ema_model), str(ckpt_path))
        print(f"checkpoint saved at step {i}")
    ckpt_path = CHECKPOINT_DIR / "ema_model.safetensors"
    safe_save(get_state_dict(ema_model), str(ckpt_path))
    print(f"saved {ckpt_path}")

  # generate and save samples from EMA model
  samples = sample_heun(ema_model, n_steps=200, bs=16)
  samples = ((samples + 1) * 127.5).clip(0, 255).cast('uint8')
  # make 4x4 grid
  grid = samples.reshape(4, 4, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(128, 128, 3)
  img = Image.fromarray(grid.numpy())
  output_path = OUTPUT_DIR / "samples.png"
  img.save(output_path)
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
