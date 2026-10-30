import argparse
import math
from pathlib import Path

import numpy as np
import pycolmap

from check_images import list_images, natural_key


# Command-line options

def parse_args():
    p = argparse.ArgumentParser(description="Decide whether an SfM result is good enough to train on.")
    p.add_argument("work", help="run_sfm.py output folder (anything containing sparse/0, sparse/1, ...)")
    p.add_argument("images", help="the image folder SfM was run on, to know which frames went missing")
    p.add_argument("--min-registered", type=float, default=0.9, help="fraction of images that must get a pose")
    p.add_argument("--max-error", type=float, default=1.0, help="warn above this mean reprojection error (px)")
    p.add_argument("--max-gap", type=float, default=30.0, help="warn if the orbit has an uncovered arc wider than this (deg)")
    p.add_argument("--plot", help="where to save the camera plot (default: <work>/sfm_report.png)")
    return p.parse_args()


# Loading

def load_models(work):
    """Every model COLMAP wrote, largest first. More than one means the images split into groups."""
    sparse = work / "sparse" if (work / "sparse").is_dir() else work
    dirs = sorted(d for d in sparse.iterdir() if d.is_dir() and d.name.isdigit())
    models = [(d.name, pycolmap.Reconstruction(d)) for d in dirs]
    return sorted(models, key=lambda m: m[1].num_reg_images(), reverse=True)


# Orbit geometry

def fit_orbit(centers, views):
    """
    Recover the orbit from the camera poses alone:
      target: the point closest to every camera's optical axis, i.e. what they were all
              looking at (least squares over the rays; this is the object's center)
      up:     normal of the plane through the camera centers (the smallest PCA direction),
              signed so the cameras look slightly down at the object
    """
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centers, views):
        P = np.eye(3) - np.outer(d, d)  # projects onto the plane perpendicular to the ray
        A += P
        b += P @ c
    target = np.linalg.solve(A, b)

    _, _, vt = np.linalg.svd(centers - centers.mean(axis=0))
    e1, e2, up = vt  # rows ordered by decreasing spread; the last one is the plane normal
    if np.mean(views @ up) > 0:
        up, e2 = -up, -e2  # keep (e1, e2, up) right-handed
    return target, e1, e2, up


def orbit_angles(centers, target, e1, e2, up):
    """Azimuth (deg) of each camera around the target, and its elevation above the target."""
    rel = centers - target
    azimuth = np.degrees(np.arctan2(rel @ e2, rel @ e1)) % 360
    elevation = np.degrees(np.arcsin((rel @ up) / np.linalg.norm(rel, axis=1)))
    return azimuth, elevation


def largest_gap(azimuth):
    """Widest arc of the orbit with no camera in it (degrees), wrapping past 360."""
    a = np.sort(azimuth)
    gaps = np.diff(np.append(a, a[0] + 360))
    i = int(np.argmax(gaps))
    return float(gaps[i]), float(a[i])


def trajectory_jumps(names, centers, views, factor=5.0, max_turn=30.0):
    """
    Frames that break the smooth path of a handheld camera. Neighboring frames are a fraction of a
    second apart, so a camera that leaps far from (or turns sharply away from) BOTH neighbors was
    placed wrong by SfM, typically a low-texture view matched to the wrong spot.
    """
    order = sorted(range(len(names)), key=lambda i: natural_key(Path(names[i])))
    c, v = centers[order], views[order]
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    turn = np.degrees(np.arccos(np.clip((v[1:] * v[:-1]).sum(1), -1, 1)))
    bad = (step > factor * np.median(step)) | (turn > max_turn)  # bad[i]: link between frame i and i+1
    return [names[order[i]] for i in range(1, len(order) - 1) if bad[i - 1] and bad[i]]


def missing_runs(all_names, registered):
    """Consecutive runs of frames that got no pose, wrapping around the orbit."""
    flags = [n not in registered for n in all_names]
    if all(flags) or not any(flags):
        return []
    start = flags.index(False)  # rotate so we begin on a registered frame
    order = all_names[start:] + all_names[:start]
    runs, cur = [], []
    for name in order:
        if name in registered:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(name)
    return sorted(runs + ([cur] if cur else []), key=len, reverse=True)


# Plot

def save_plot(path, rec, centers, views, azimuth, target, e1, e2, up, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radius = np.median(np.linalg.norm(centers - target, axis=1))
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cols = np.array([p.color for p in rec.points3D.values()]) / 255.0
    near = np.linalg.norm(pts - target, axis=1) < 1.5 * radius  # drop far background for readability

    fig, (top, seq) = plt.subplots(1, 2, figsize=(13, 6))
    rel_pts, rel_cams = pts[near] - target, centers - target
    top.scatter(rel_pts @ e1, rel_pts @ e2, s=1, c=cols[near])
    top.scatter(rel_cams @ e1, rel_cams @ e2, s=12, c="tab:red", label="cameras")
    top.quiver(rel_cams @ e1, rel_cams @ e2, views @ e1, views @ e2, color="tab:red", width=0.003,
               scale=1 / (0.15 * radius), scale_units="xy")
    top.scatter([0], [0], marker="x", s=80, c="k", label="look-at target")
    top.set_aspect("equal")
    top.set_title("Top view (looking down the orbit axis)")
    top.legend(loc="upper right")

    # In capture order, a clean orbit is a straight ramp of azimuth vs frame number.
    # Jumps mean frames out of order or a camera placed in the wrong spot.
    idx = [names.index(n) for n in sorted(names)]
    travelled = np.degrees(np.unwrap(np.radians(azimuth[idx])))
    seq.plot(travelled - travelled[0], ".-")
    seq.set_xlabel("registered frame (capture order)")
    seq.set_ylabel("degrees travelled around target")
    seq.set_title("Orbit progress")
    seq.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Main

def main():
    args = parse_args()
    work = Path(args.work)
    all_names = [p.name for p in list_images(args.images)]
    models = load_models(work)
    if not models:
        raise SystemExit(f"FAIL: no SfM models found under {work}")

    errors, warnings = [], []
    model_id, rec = models[0]

    # Registration: how many frames got a pose, and whether the missing ones are clustered.
    reg = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}

    # Extra models matter only if they hold frames the main one lacks. COLMAP sometimes leaves behind
    # a tiny model from a failed first attempt whose frames were all re-registered in the main model.
    lost = {m.images[i].name for _, m in models[1:] for i in m.reg_image_ids()} - set(reg)
    if lost:
        sizes = ", ".join(f"sparse/{i}: {m.num_reg_images()}" for i, m in models)
        warnings.append(f"{len(models)} separate models ({sizes}); {len(lost)} frames exist only outside "
                        f"sparse/{model_id}, so that part of the orbit is lost")
    frac = len(reg) / len(all_names)
    if frac < args.min_registered:
        errors.append(f"only {len(reg)}/{len(all_names)} images registered ({frac:.0%} < {args.min_registered:.0%})")
    runs = missing_runs(all_names, reg)
    if runs:
        warnings.append(f"unregistered frames: {sum(map(len, runs))}, longest run {len(runs[0])} "
                        f"({runs[0][0]} .. {runs[0][-1]})")

    # Accuracy: mean distance between where points reproject and where SIFT found them.
    err = rec.compute_mean_reprojection_error()
    if err > args.max_error:
        warnings.append(f"mean reprojection error {err:.2f}px > {args.max_error}px (blur, rolling shutter, wrong camera model?)")

    # Intrinsics: a phone's main camera sees roughly 55-80 degrees across the long side of the frame
    # (measured on the long side so portrait and landscape video are judged the same way).
    cam = next(iter(rec.cameras.values()))
    if cam.width >= cam.height:
        fov = math.degrees(2 * math.atan(cam.width / (2 * cam.focal_length_x)))
    else:
        fov = math.degrees(2 * math.atan(cam.height / (2 * cam.focal_length_y)))
    if not 40 <= fov <= 100:
        warnings.append(f"FOV {fov:.0f} deg (long side) looks implausible for a phone; the focal length may be wrong")

    # Orbit coverage: angles are measured from the recovered poses, not assumed from frame order.
    names = list(reg)
    centers = np.array([reg[n].projection_center() for n in names])
    views = np.array([reg[n].viewing_direction() for n in names])
    target, e1, e2, up = fit_orbit(centers, views)
    azimuth, elevation = orbit_angles(centers, target, e1, e2, up)
    gap, gap_start = largest_gap(azimuth)
    if gap > args.max_gap:
        warnings.append(f"no cameras between azimuth {gap_start:.0f} and {(gap_start + gap) % 360:.0f} deg "
                        f"({gap:.0f} deg gap): that side of the object will be poorly reconstructed")

    # Misplaced cameras: registered, so the counts above look fine, but wrong. A wrong pose is worse
    # than a missing one: training tries to explain that photo from the wrong spot and grows floaters.
    jumps = trajectory_jumps(names, centers, views)
    if jumps:
        errors.append(f"{len(jumps)} camera(s) jump away from both neighbors, so SfM likely misplaced them: "
                      f"{', '.join(jumps)}. Delete those frames and re-run run_sfm.py")

    print(f"Models: {len(models)}   using sparse/{model_id}")
    print(f"Registered: {len(reg)}/{len(all_names)} ({frac:.0%})")
    print(f"3D points: {rec.num_points3D()}   mean track length {rec.compute_mean_track_length():.1f}   "
          f"observations/image {rec.compute_mean_observations_per_reg_image():.0f}")
    print(f"Reprojection error: {err:.3f}px")
    print(f"Camera: {cam.model.name} {cam.width}x{cam.height}, fx {cam.focal_length_x:.0f}px "
          f"(FOV {fov:.0f} deg on the long side), params {np.round(cam.params, 4).tolist()}")
    print(f"Orbit: largest gap {gap:.0f} deg, elevation {elevation.min():.0f} to {elevation.max():.0f} deg "
          f"(median {np.median(elevation):.0f})")

    plot = Path(args.plot) if args.plot else work / "sfm_report.png"
    save_plot(plot, rec, centers, views, azimuth, target, e1, e2, up, names)
    print(f"Plot: {plot}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("\nFAIL: do not train on this" if errors else
          "\nOK to train" + (" (review warnings above)" if warnings else ""))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
