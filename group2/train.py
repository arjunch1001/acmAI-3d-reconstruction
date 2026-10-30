import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

SH_C0 = 0.28209479177387814  # constant of the degree-0 spherical harmonic: color = SH_C0 * sh0 + 0.5


# Command-line options

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a 3D Gaussian Splat on a run_sfm.py dataset with gsplat.")
    p.add_argument("data", help="run_sfm.py dataset folder (images/ + sparse/0, undistorted pinhole)")
    p.add_argument("out", help="output folder for splat.ply, checkpoint, metrics, previews")
    p.add_argument("--steps", type=int, default=3000, help="training iterations (Inria's full run is 30000)")
    p.add_argument("--strategy", choices=["default", "mcmc"], default="default",
                   help="default = Inria clone/split/prune; mcmc = fixed Gaussian budget, no opacity resets, "
                        "but it grows only 5%% per refine round, so it suits long runs, not the 3k-step one")
    p.add_argument("--refine-stop", type=float, default=0.7,
                   help="stop adding/removing Gaussians after this fraction of steps, so the rest only refines")
    p.add_argument("--reset-every", type=int, default=0,
                   help="opacity reset period in steps (0 = never; Inria uses 3000 in a 30000-step run)")
    p.add_argument("--prune-opa", type=float, default=0.005,
                   help="delete Gaussians whose opacity falls below this (the spec's pruning threshold)")
    p.add_argument("--cap-max", type=int, default=500_000, help="mcmc: maximum number of Gaussians")
    p.add_argument("--sh-degree", type=int, default=3, help="max spherical-harmonic degree (view-dependent color)")
    p.add_argument("--downscale", type=int, default=1, help="train on images shrunk by this factor")
    p.add_argument("--holdout", type=int, default=0,
                   help="hold out every Nth image to measure PSNR on unseen views (8 is the usual choice)")
    p.add_argument("--no-random-bg", action="store_true", help="render onto black instead of a random color")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--preview-every", type=int, default=500, help="save a render-vs-photo image this often")
    p.add_argument("--save-every", type=int, default=1000, help="checkpoint this often (Colab can disconnect)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# Data

def load_dataset(data, downscale, holdout):
    """Cameras, photos, and the SfM point cloud from a COLMAP model."""
    rec = pycolmap.Reconstruction(data / "sparse" / "0")
    views = []
    for img_id in sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name):
        image = rec.images[img_id]
        cam = rec.cameras[image.camera_id]
        if cam.model.name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise SystemExit(f"camera model {cam.model.name}: train on run_sfm.py's undistorted dataset folder")
        pixels = cv2.imread(str(data / "images" / image.name))[..., ::-1]  # OpenCV loads BGR
        h, w = pixels.shape[:2]
        if downscale > 1:
            pixels = cv2.resize(pixels, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
        sx, sy = pixels.shape[1] / w, pixels.shape[0] / h  # intrinsics scale with the image
        K = np.array([[cam.focal_length_x * sx, 0, cam.principal_point_x * sx],
                      [0, cam.focal_length_y * sy, cam.principal_point_y * sy],
                      [0, 0, 1]])
        viewmat = np.eye(4)
        viewmat[:3] = image.cam_from_world().matrix()  # world -> camera, OpenCV axes (what gsplat expects)
        views.append({"name": image.name, "pixels": np.ascontiguousarray(pixels), "K": K, "viewmat": viewmat,
                      "center": image.projection_center()})

    is_test = [bool(holdout) and i % holdout == 0 for i in range(len(views))]
    test = [v for v, t in zip(views, is_test) if t]
    train = [v for v, t in zip(views, is_test) if not t]
    points = np.array([p.xyz for p in rec.points3D.values()])
    colors = np.array([p.color for p in rec.points3D.values()]) / 255.0

    # Scene size from how far the cameras spread: learning rates and size thresholds are relative to it,
    # so the same settings work whatever units SfM happened to pick.
    centers = np.array([v["center"] for v in views])
    scene_scale = 1.1 * float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
    return train, test, points, colors, scene_scale


def to_device(views, device):
    """Photos stay uint8 on the GPU (4x smaller than float); converted one at a time when used."""
    return {
        "pixels": torch.from_numpy(np.stack([v["pixels"] for v in views])).to(device),
        "K": torch.tensor(np.stack([v["K"] for v in views]), dtype=torch.float32, device=device),
        "viewmat": torch.tensor(np.stack([v["viewmat"] for v in views]), dtype=torch.float32, device=device),
    }


# Model

def mean_neighbor_distance(points, k=3, chunk=4096):
    """Average distance from each point to its k nearest neighbors, in chunks to bound memory."""
    out = []
    for s in range(0, len(points), chunk):
        d = torch.cdist(points[s:s + chunk], points)
        out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(1))  # column 0 is the point itself
    return torch.cat(out)


def init_gaussians(points, colors, args, scene_scale, device):
    """
    One Gaussian per SfM point. Each parameter is stored in a space where any real number is valid
    (log scale, logit opacity), so the optimizer can never push it to an impossible value.
    """
    means = torch.tensor(points, dtype=torch.float32, device=device)
    rgb = torch.tensor(colors, dtype=torch.float32, device=device)
    n = len(means)
    dist = mean_neighbor_distance(means).clamp_min(1e-7)  # sized to touch their neighbors
    opacity = 0.5 if args.strategy == "mcmc" else 0.1     # mcmc's published setting vs Inria's
    n_rest = (args.sh_degree + 1) ** 2 - 1

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), opacity, device=device))),
        "sh0": torch.nn.Parameter(((rgb - 0.5) / SH_C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, n_rest, 3, device=device)),
    })
    # Inria's learning rates. Positions move in scene units, so their rate scales with the scene;
    # higher SH bands get a 20x smaller rate so view-dependent color can't explain away geometry errors.
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    return params, optimizers


def make_strategy(args):
    """The densification schedule, scaled to a short run (see the comments on each value)."""
    refine_start = min(500, args.steps // 6)       # let gradients settle before judging which areas need detail
    refine_stop = int(args.refine_stop * args.steps)
    if args.strategy == "mcmc":
        return MCMCStrategy(cap_max=args.cap_max, refine_start_iter=refine_start, refine_stop_iter=refine_stop,
                            min_opacity=args.prune_opa)
    # A reset period past the end of training disables resets. A reset shortly before the end would
    # leave every Gaussian half-transparent in the exported file.
    reset_every = args.reset_every if args.reset_every > 0 else args.steps + 1
    return DefaultStrategy(prune_opa=args.prune_opa, refine_start_iter=refine_start,
                           refine_stop_iter=refine_stop, reset_every=reset_every)


# Rendering and loss

def render(params, data, idx, sh_degree, background):
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],                       # normalized inside gsplat
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.cat([params["sh0"], params["shN"]], 1),
        viewmats=data["viewmat"][idx:idx + 1],
        Ks=data["K"][idx:idx + 1],
        width=data["pixels"].shape[2],
        height=data["pixels"].shape[1],
        sh_degree=sh_degree,
        backgrounds=background[None],
        packed=False,
    )
    return out[0], info


def ssim(a, b, window=11, sigma=1.5):
    """Structural similarity of two [H, W, 3] images: compares local brightness, contrast, and structure."""
    x = torch.arange(window, device=a.device, dtype=a.dtype) - window // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).expand(3, 1, window, window)
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    blur = lambda t: F.conv2d(t, kernel, padding=window // 2, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean()


def psnr(a, b):
    return float(-10 * torch.log10(F.mse_loss(a, b)))


# Saving

def save_ply(params, path):
    """
    Standard 3DGS .ply (the Inria layout every splat viewer reads). Values are saved raw, before
    their activations: viewers apply exp() to scales and sigmoid() to opacity themselves.
    """
    n = len(params["means"])
    f_rest = params["shN"].transpose(1, 2).reshape(n, -1)  # channel-major: all R coeffs, then G, then B
    columns = {
        "x": params["means"][:, 0], "y": params["means"][:, 1], "z": params["means"][:, 2],
        **{f"n{a}": torch.zeros(n) for a in "xyz"},        # normals: unused, but part of the layout
        **{f"f_dc_{i}": params["sh0"][:, 0, i] for i in range(3)},
        **{f"f_rest_{i}": f_rest[:, i] for i in range(f_rest.shape[1])},
        "opacity": params["opacities"],
        **{f"scale_{i}": params["scales"][:, i] for i in range(3)},
        **{f"rot_{i}": F.normalize(params["quats"], dim=-1)[:, i] for i in range(4)},
    }
    table = np.empty(n, dtype=[(k, "<f4") for k in columns])
    for k, v in columns.items():
        table[k] = v.detach().cpu().numpy()
    header = "".join(["ply\nformat binary_little_endian 1.0\n", f"element vertex {n}\n",
                      *[f"property float {k}\n" for k in columns], "end_header\n"])
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(table.tobytes())


def save_preview(params, data, idx, sh_degree, path):
    with torch.no_grad():
        img, _ = render(params, data, idx, sh_degree, torch.zeros(3, device=data["K"].device))
    photo = data["pixels"][idx].cpu().numpy()
    pred = (img.clamp(0, 1) * 255).byte().cpu().numpy()
    cv2.imwrite(str(path), np.hstack([photo, pred])[..., ::-1])  # left: photo, right: render


@torch.no_grad()
def evaluate(params, data, sh_degree):
    if len(data["pixels"]) == 0:
        return None
    black = torch.zeros(3, device=data["K"].device)
    scores = [psnr(render(params, data, i, sh_degree, black)[0].clamp(0, 1), data["pixels"][i].float() / 255)
              for i in range(len(data["pixels"]))]
    return float(np.mean(scores))


# Training loop

def train(args, device):
    torch.manual_seed(args.seed)
    data_dir, out = Path(args.data), Path(args.out)
    (out / "previews").mkdir(parents=True, exist_ok=True)

    train_views, test_views, points, colors, scene_scale = load_dataset(data_dir, args.downscale, args.holdout)
    train_data, test_data = to_device(train_views, device), to_device(test_views, device)
    h, w = train_views[0]["pixels"].shape[:2]
    print(f"{len(train_views)} training + {len(test_views)} held-out views at {w}x{h}, "
          f"{len(points)} SfM points, scene scale {scene_scale:.2f}")

    params, optimizers = init_gaussians(points, colors, args, scene_scale, device)
    strategy = make_strategy(args)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state() if args.strategy == "mcmc" else strategy.initialize_state(scene_scale)
    # Position learning rate decays 100x over the run: big moves early, fine placement at the end.
    means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / args.steps))
    # Unlock one more SH band at evenly spaced steps, reaching full degree halfway through:
    # color first, then view-dependent effects once geometry is roughly right.
    sh_every = max(1, args.steps // (2 * args.sh_degree)) if args.sh_degree else args.steps

    log_file = open(out / "metrics.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "loss", "psnr", "gaussians", "seconds"])
    preview_idx = 0
    preview_data = test_data if len(test_views) else train_data
    order, start, recent = [], time.perf_counter(), []

    for step in range(args.steps):
        if not order:  # new epoch: visit every training view once, in random order
            order = torch.randperm(len(train_views)).tolist()
        idx = order.pop()
        sh_degree = min(step // sh_every, args.sh_degree)
        background = (torch.zeros(3, device=device) if args.no_random_bg else torch.rand(3, device=device))

        pred, info = render(params, train_data, idx, sh_degree, background)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        photo = train_data["pixels"][idx].float() / 255
        loss = 0.8 * (pred - photo).abs().mean() + 0.2 * (1 - ssim(pred, photo))
        if args.strategy == "mcmc":  # the regularizers MCMC relies on to retire useless Gaussians
            loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_decay.step()
        if args.strategy == "mcmc":
            strategy.step_post_backward(params, optimizers, state, step, info, lr=means_decay.get_last_lr()[0])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        recent.append((loss.item(), psnr(pred.detach(), photo)))
        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            elapsed = time.perf_counter() - start
            l, p = np.mean(recent, axis=0)
            recent = []
            n = len(params["means"])
            log.writerow([done, f"{l:.5f}", f"{p:.2f}", n, f"{elapsed:.1f}"])
            log_file.flush()
            eta = elapsed / done * (args.steps - done)
            print(f"step {done:>6}/{args.steps}  loss {l:.4f}  train PSNR {p:5.2f}  gaussians {n:>8,}  "
                  f"SH {sh_degree}  {done / elapsed:5.1f} it/s  ETA {eta / 60:4.1f} min", flush=True)
        if done % args.preview_every == 0 or done == args.steps:
            save_preview(params, preview_data, preview_idx, sh_degree, out / "previews" / f"step_{done:06d}.png")
        if done % args.save_every == 0 or done == args.steps:
            torch.save({"step": done, "params": params.state_dict(), "args": vars(args)}, out / "checkpoint.pt")

    log_file.close()
    elapsed = time.perf_counter() - start
    save_ply(params, out / "splat.ply")
    summary = {"steps": args.steps, "minutes": round(elapsed / 60, 2), "gaussians": len(params["means"]),
               "test_psnr": evaluate(params, test_data, args.sh_degree), "train_psnr": evaluate(params, train_data, args.sh_degree),
               "ply_mb": round((out / "splat.ply").stat().st_size / 1e6, 1), "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {elapsed / 60:.1f} min: {summary['gaussians']:,} Gaussians, {summary['ply_mb']} MB, "
          f"train PSNR {summary['train_psnr']:.2f}"
          + (f", held-out PSNR {summary['test_psnr']:.2f}" if summary["test_psnr"] is not None else ""))
    print(f"Splat: {out / 'splat.ply'}")
    return summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. gsplat needs an NVIDIA GPU: run this on Colab (see pipeline.ipynb).")
    train(args, "cuda")


if __name__ == "__main__":
    main()
