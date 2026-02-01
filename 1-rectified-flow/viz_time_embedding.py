import math
import torch
import matplotlib.pyplot as plt

def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
  args = t[:, None] * freqs[None]
  return torch.cat([args.sin(), args.cos()], dim=1)

# 1) Heatmap over time
T = 400
dim = 64
t = torch.linspace(0, 50, T)
emb = timestep_embedding(t, dim=dim).T  # [dim, T]

plt.figure()
plt.imshow(emb.numpy(), aspect='auto', origin='lower')
plt.xlabel('time index')
plt.ylabel('embedding dimension')
plt.title('timestep embedding heatmap')
plt.colorbar()

# 2) A few channels over time
plt.figure()
half = dim // 2
ks = [0, half // 4, half // 2, half - 1]
for k in ks:
  plt.plot(t.numpy(), emb[k].numpy(), label=f"sin k={k}")
  plt.plot(t.numpy(), emb[half + k].numpy(), linestyle='--', label=f"cos k={k}")
plt.xlabel('t')
plt.title('selected sin/cos channels')
plt.legend(ncol=2, fontsize=8)

# 3) Circle plot for one frequency pair
k = half // 2
x = emb[half + k]   # cos
y = emb[k]          # sin
plt.figure()
plt.plot(x.numpy(), y.numpy())
plt.gca().set_aspect('equal', 'box')
plt.xlabel('cos')
plt.ylabel('sin')
plt.title(f'unit-circle trajectory for k={k}')

plt.show()
plt.show()
