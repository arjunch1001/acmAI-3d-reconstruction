"""Confirm gsplat's prebuilt GPU code loads and renders on this machine, before spending time on training."""
import time

import torch
import gsplat
from gsplat import rasterization

assert torch.cuda.is_available(), "No CUDA GPU: Runtime -> Change runtime type -> T4 GPU"
start = time.perf_counter()
n = 10_000
means = torch.randn(n, 3, device="cuda") + torch.tensor([0, 0, 5.0], device="cuda")
quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
img, _, _ = rasterization(means, quats, torch.full((n, 3), 0.05, device="cuda"), torch.full((n,), 0.8, device="cuda"),
                          torch.rand(n, 3, device="cuda"), torch.eye(4, device="cuda")[None],
                          torch.tensor([[[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]], device="cuda"), 640, 480)
torch.cuda.synchronize()
print(f"OK: gsplat {gsplat.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}, "
      f"test render {tuple(img.shape)} in {time.perf_counter() - start:.1f}s")
