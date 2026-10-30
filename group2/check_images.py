import argparse
import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# Command-line options

def parse_args():
    p = argparse.ArgumentParser(description="Check Group 1's image folder before spending GPU time on SfM.")
    p.add_argument("images", help="folder of frames from Group 1")
    p.add_argument("--min-count", type=int, default=80, help="fewest frames the spec allows")
    p.add_argument("--max-count", type=int, default=130, help="most frames the spec allows")
    p.add_argument("--long-edge", type=int, default=1280, help="expected longest image side in px")
    p.add_argument("--blur", type=float, default=0.5,
                   help="flag images sharper than less than this fraction of the folder's median")
    p.add_argument("--min-features", type=int, default=500, help="flag images with fewer SIFT keypoints")
    p.add_argument("--min-inliers", type=int, default=100,
                   help="flag neighboring frames sharing fewer geometrically verified matches")
    p.add_argument("--resize-to", help="write copies downscaled to --long-edge into this folder, then check those")
    return p.parse_args()


# Loading

def natural_key(path):
    """Sort frame_2 before frame_10 (plain string sort would not)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def list_images(folder):
    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def downscale(img, long_edge):
    """Shrink so the longest side is long_edge px. Never upscale: that adds pixels, not detail."""
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    if s >= 1:
        return img
    # INTER_AREA averages the source pixels each output pixel covers, which avoids aliasing when shrinking.
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


# The per-image and per-pair checks

def sharpness(gray):
    """Variance of the Laplacian: edges are strong in sharp images, weak in blurry ones."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def verified_inliers(a, b, matcher, ratio=0.8, ransac_px=4.0):
    """
    How many keypoints two images share, counted the way COLMAP does it:
    nearest-neighbor descriptor matches that pass Lowe's ratio test, then only the ones
    consistent with a single epipolar geometry (RANSAC on the fundamental matrix).
    """
    (kp_a, desc_a), (kp_b, desc_b) = a, b
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 8:  # the 8-point algorithm needs at least 8 correspondences
        return 0
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, ransac_px, 0.999)
    return int(mask.sum()) if mask is not None else 0


# Main

def main():
    args = parse_args()
    paths = list_images(args.images)
    if not paths:
        raise SystemExit(f"No images found in {args.images}")

    out_dir = Path(args.resize_to) if args.resize_to else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    sift = cv2.SIFT_create(nfeatures=8192)  # same cap as COLMAP's default max_num_features
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    errors, warnings = [], []
    sizes = {}                       # (w, h) -> image names, to catch mixed resolutions
    sharp, feats, links = [], [], []
    first = prev = None              # (name, (keypoints, descriptors)); only two kept so memory stays flat

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            errors.append(f"{path.name}: unreadable or corrupt")
            continue
        if out_dir:
            img = downscale(img, args.long_edge)
            cv2.imwrite(str(out_dir / f"{path.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        h, w = img.shape[:2]
        sizes.setdefault((w, h), []).append(path.name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp.append((path.name, sharpness(gray)))
        kp, desc = sift.detectAndCompute(gray, None)
        feats.append((path.name, len(kp)))

        cur = (path.name, (kp, desc))
        if prev:
            links.append((prev[0], cur[0], verified_inliers(prev[1], cur[1], matcher)))
        first = first or cur
        prev = cur
        print(f"\r  analyzed {len(feats)}/{len(paths)}", end="", flush=True)
    print()

    # Count: the spec's range, not a hard SfM limit.
    n = len(feats)
    if not args.min_count <= n <= args.max_count:
        warnings.append(f"{n} readable images; spec wants {args.min_count}-{args.max_count}")

    # Resolution: COLMAP will treat every frame as the same camera, so sizes must match.
    if len(sizes) > 1:
        detail = ", ".join(f"{w}x{h} ({len(v)} imgs)" for (w, h), v in sizes.items())
        errors.append(f"mixed resolutions: {detail}. All frames must come from one camera setting")
    for (w, h) in sizes:
        if max(w, h) != args.long_edge:
            warnings.append(f"longest edge is {max(w, h)}px, expected {args.long_edge} (fix with --resize-to)")

    bad = set()  # frames already flagged, so their weak links aren't reported as orbit breaks

    # Blur: relative to this folder's median, since absolute sharpness depends on the object's texture.
    if sharp:
        median = float(np.median([s for _, s in sharp]))
        for name, s in sharp:
            if s < args.blur * median:
                warnings.append(f"{name}: blurry ({s:.0f} vs median {median:.0f})")
                bad.add(name)

    # Texture: SfM can only place a camera if it sees enough distinctive keypoints.
    for name, k in feats:
        if k < args.min_features:
            warnings.append(f"{name}: only {k} SIFT keypoints (low texture?)")
            bad.add(name)

    # Overlap: a weak link between neighbors is where COLMAP's chain of registered images can snap.
    # Two weak links around one frame mean that frame is the problem (blurry, occluded), not the orbit.
    weak = [inl < args.min_inliers for *_, inl in links]
    for i, (a, b, inl) in enumerate(links):
        if not weak[i] or a in bad or b in bad:
            continue
        if i + 1 < len(links) and weak[i + 1]:
            warnings.append(f"{b}: weakly matched on both sides ({inl}, {links[i + 1][2]}); "
                            "COLMAP will likely drop it, consider deleting it")
        elif not (i > 0 and weak[i - 1]):
            warnings.append(f"{a} -> {b}: only {inl} verified matches; the orbit may break here")

    # Loop closure is informational: a full orbit should end looking roughly where it started.
    loop = verified_inliers(prev[1], first[1], matcher) if n > 2 else 0

    print(f"\nImages: {n}   sizes: {', '.join(f'{w}x{h}' for w, h in sizes)}")
    if feats:
        print(f"SIFT keypoints per image: median {int(np.median([k for _, k in feats]))}, "
              f"min {min(k for _, k in feats)}")
    if links:
        print(f"Neighbor verified matches: median {int(np.median([i for *_, i in links]))}, "
              f"min {min(i for *_, i in links)}")
    print(f"Loop closure (last -> first): {loop} verified matches")
    if out_dir:
        print(f"Resized copies written to {out_dir}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("\nFAIL: fix the errors before running SfM" if errors else
          "\nOK to run SfM" + (" (review warnings above)" if warnings else ""))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
