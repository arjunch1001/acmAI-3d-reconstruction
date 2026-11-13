import argparse
import gzip
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814  # degree-0 spherical harmonic: rgb = 0.5 + SH_C0 * f_dc

# Approximate average texture colors (sRGB) of full, opaque, plain-looking blocks.
# Concrete, wool and terracotta give 49 colors with flat textures, so a build reads as clean color.
BLOCKS = {
    "white_concrete": (207, 213, 214), "orange_concrete": (224, 97, 1), "magenta_concrete": (169, 48, 159),
    "light_blue_concrete": (36, 137, 199), "yellow_concrete": (241, 175, 21), "lime_concrete": (94, 169, 24),
    "pink_concrete": (214, 101, 143), "gray_concrete": (55, 58, 62), "light_gray_concrete": (125, 125, 115),
    "cyan_concrete": (21, 119, 136), "purple_concrete": (100, 32, 156), "blue_concrete": (45, 47, 143),
    "brown_concrete": (96, 60, 32), "green_concrete": (73, 91, 36), "red_concrete": (142, 33, 33),
    "black_concrete": (8, 10, 15),
    "white_wool": (234, 236, 237), "orange_wool": (241, 118, 20), "magenta_wool": (190, 69, 180),
    "light_blue_wool": (58, 175, 217), "yellow_wool": (249, 198, 40), "lime_wool": (112, 185, 26),
    "pink_wool": (238, 141, 172), "gray_wool": (63, 68, 72), "light_gray_wool": (142, 142, 135),
    "cyan_wool": (21, 138, 145), "purple_wool": (122, 42, 173), "blue_wool": (53, 57, 157),
    "brown_wool": (114, 72, 41), "green_wool": (85, 110, 28), "red_wool": (161, 39, 35), "black_wool": (21, 21, 26),
    "terracotta": (152, 94, 68), "white_terracotta": (210, 178, 161), "orange_terracotta": (162, 84, 38),
    "magenta_terracotta": (150, 88, 109), "light_blue_terracotta": (113, 109, 138),
    "yellow_terracotta": (186, 133, 35), "lime_terracotta": (104, 118, 53), "pink_terracotta": (162, 78, 79),
    "gray_terracotta": (58, 42, 36), "light_gray_terracotta": (135, 107, 98), "cyan_terracotta": (87, 91, 91),
    "purple_terracotta": (118, 70, 86), "blue_terracotta": (74, 60, 91), "brown_terracotta": (77, 51, 36),
    "green_terracotta": (76, 83, 42), "red_terracotta": (143, 61, 47), "black_terracotta": (37, 23, 17),
}


# Command-line options

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Turn a Gaussian Splat .ply into a Minecraft .schem (WorldEdit) build.")
    p.add_argument("ply", help="Gaussian Splat .ply (train.py output, or any Inria-layout 3DGS file)")
    p.add_argument("out", help="output .schem")
    p.add_argument("--cameras", help="SfM cameras in the splat's coordinates: run_sfm.py's dataset/sparse/0, "
                                     "or a JSON of {name: {R, t, K, size}}. Used to find the object and which way is up")
    p.add_argument("--size", type=int, default=64, help="blocks along the object's longest side")
    p.add_argument("--min-visible", type=float, default=0.5,
                   help="keep only Gaussians inside the frame of at least this fraction of cameras (the object)")
    p.add_argument("--no-ground-removal", action="store_true",
                   help="keep the surface the object stands on (stool, table, floor) and everything below it")
    p.add_argument("--min-opacity", type=float, default=0.01,
                   help="ignore Gaussians more transparent than this (faint ones still add up into solid surfaces)")
    p.add_argument("--density", type=float, default=0.5,
                   help="opacity a voxel must collect to become a block (1 = one fully opaque Gaussian)")
    p.add_argument("--close", type=int, default=1, help="seal holes up to about 2x this many blocks wide (0 = off)")
    p.add_argument("--fill", action="store_true", help="fill the hollow inside of the object with blocks")
    p.add_argument("--no-smooth", action="store_true", help="don't average colors with neighboring blocks")
    p.add_argument("--up", help="up direction as x,y,z if there are no cameras (default 0,-1,0: COLMAP's usual)")
    p.add_argument("--data-version", type=int, default=3465,
                   help="Minecraft data version the file claims (3465 = Java 1.20.1); newer WorldEdit upgrades it")
    p.add_argument("--preview", help="save a PNG of the build from four sides (default: next to the .schem)")
    return p.parse_args(argv)


# Reading

def read_ply(path):
    """Binary little-endian .ply with one 'property <type> <name>' per column, read using its own header."""
    with open(path, "rb") as f:
        header = []
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            header.append(line)
        if "format binary_little_endian 1.0" not in header:
            raise SystemExit(f"{path}: only binary little-endian .ply files are supported")
        count = next(int(l.split()[2]) for l in header if l.startswith("element vertex"))
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
        fields = [(l.split()[2], types[l.split()[1]]) for l in header if l.startswith("property")]
        v = np.frombuffer(f.read(), dtype=fields, count=count)
    # Undo the storage transforms: the file keeps log-scale, logit-opacity and SH coefficients.
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    rgb = np.clip(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return means, scales, alpha, rgb, quats


def load_cameras(path):
    """List of (R, t, K, (width, height)) with x_cam = R @ x_world + t (OpenCV axes: x right, y down, z forward)."""
    path = Path(path)
    if path.suffix == ".json":
        cams = json.load(open(path)).values()
        return [(np.array(c["R"]), np.array(c["t"]), np.array(c["K"]), tuple(c["size"])) for c in cams]
    import pycolmap
    rec = pycolmap.Reconstruction(path)
    out = []
    for i in rec.reg_image_ids():
        pose, cam = rec.images[i].cam_from_world(), rec.cameras[rec.images[i].camera_id]
        out.append((pose.rotation.matrix(), np.asarray(pose.translation), cam.calibration_matrix(),
                    (cam.width, cam.height)))
    return out


# Finding the object

def visible_fraction(points, cams, margin=0.05):
    """
    Fraction of cameras that have each point inside their frame (ignoring occlusion).
    In an orbit every camera aims at the object, so object points score near 1, while walls and
    far floor only come into frame when the camera happens to face them.
    """
    hits = np.zeros(len(points))
    for R, t, K, (w, h) in cams:
        pc = points @ R.T + t
        z = pc[:, 2]
        uv = (pc @ K.T)[:, :2] / np.maximum(z, 1e-9)[:, None]
        hits += ((z > 0) & (uv[:, 0] > margin * w) & (uv[:, 0] < (1 - margin) * w)
                 & (uv[:, 1] > margin * h) & (uv[:, 1] < (1 - margin) * h))
    return hits / len(cams)


def camera_up(cams):
    """Average of each camera's 'up' (-y in OpenCV) in world coordinates: the person held the phone upright."""
    up = np.mean([-R[1] for R, _, _, _ in cams], axis=0)
    return up / np.linalg.norm(up)


def rotation_to_y(up):
    """Rotation matrix taking `up` to +Y (Minecraft's up). A rotation, never a mirror: mirroring would turn a left shoe into a right one."""
    a, b = up / np.linalg.norm(up), np.array([0.0, 1.0, 0.0])
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)  # Rodrigues' formula


def fit_plane(points, weights, up, thresh, rng, iters=500, max_tilt=35):
    """
    RANSAC: repeatedly fit a plane through 3 random points and keep the one most points lie on.
    Only near-horizontal planes (normal within max_tilt of `up`) count. Returns (normal, offset, inliers),
    normal pointing up, or None.
    """
    p = weights / weights.sum()
    best, best_score = None, 0.0
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False, p=p)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ up > 0 else -n
        if n @ up < np.cos(np.radians(max_tilt)):
            continue
        inl = np.abs((points - a) @ n) < thresh
        score = weights[inl].sum()
        if score > best_score:
            best, best_score = (n, float(a @ n)), score
    if best is None:
        return None
    # Refine with a least-squares fit to all inliers (the 3-point fit is only as good as its 3 points).
    n, d = best
    inl = np.abs(points @ n - d) < thresh
    q = points[inl] - points[inl].mean(0)
    n = np.linalg.svd(q * weights[inl, None], full_matrices=False)[2][2]
    n = n if n @ up > 0 else -n
    return n, float(points[inl].mean(0) @ n), np.abs(points @ n - points[inl].mean(0) @ n) < thresh


def support_plane(points, weights, up, rng, max_planes=3, min_share=0.03, min_above=0.1, wider=1.1):
    """
    The surface the object stands on (stool seat, table, floor). Pull out up to `max_planes` big
    horizontal planes and pick the biggest one that has enough above it to be the object and sticks
    out further than what's on top of it. The object itself has flat-ish horizontal slices too, but
    they're smaller and don't stick out past the object. Tested on uCO3D: the seat was always biggest.
    """
    extent = np.linalg.norm(np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0))
    thresh = 0.01 * extent
    total, free, best = weights.sum(), np.ones(len(points), bool), None
    for _ in range(max_planes):
        if free.sum() < 3:
            break
        fit = fit_plane(points[free], weights[free], up, thresh, rng, max_tilt=20)  # camera 'up' is off by <10 deg
        if fit is None:
            break
        n, d, _ = fit
        dist = points @ n - d
        on = np.abs(dist) < thresh
        free &= ~on
        if weights[on].sum() < min_share * total:
            break
        above = dist > 2 * thresh
        if weights[above].sum() < min_above * total:
            continue
        center = points[above].mean(0)
        radial = lambda m: np.linalg.norm((points[m] - center) - np.outer((points[m] - center) @ n, n), axis=1)
        if np.percentile(radial(on), 90) < wider * np.percentile(radial(above), 90):
            continue
        if best is None or weights[on].sum() > best[2]:
            best = (n, d, weights[on].sum(), thresh)
    return None if best is None else (best[0], best[1], best[3])


def find_object(means, alpha, cams, args, rng):
    """
    Which Gaussians are the object, and which way is up.
    1. Cameras: an orbit keeps the object in frame the whole time; walls and far floor drift in and out.
    2. Support: cut along the plane the object stands on, and everything under it (legs, floor).
       The plane's normal is also the true 'up': averaging camera tilts is only approximate.
    """
    keep = alpha >= args.min_opacity
    if cams:
        keep &= visible_fraction(means, cams) >= args.min_visible
        up = camera_up(cams)
        print(f"{len(cams)} cameras: {keep.sum():,} Gaussians stay in frame for >= {args.min_visible:.0%} of them")
    else:
        up = np.array([float(v) for v in (args.up or "0,-1,0").split(",")])
        up /= np.linalg.norm(up)
        print("No cameras given: keeping everything; the largest connected blob will be taken as the object")
    if not args.no_ground_removal:
        sure = keep & (alpha >= 0.1)  # find the plane with clearly visible Gaussians; faint ones blur it
        plane = support_plane(means[sure], alpha[sure], up, rng)
        if plane is None:
            print("No support surface found under the object")
        else:
            n, d, thresh = plane
            tilt = np.degrees(np.arccos(np.clip(n @ up, -1, 1)))
            before = keep.sum()
            keep &= means @ n - d > thresh  # 1x the plane's thickness: keeps the sole, drops the seat (tuned on uCO3D)
            print(f"Cut along the support surface: removed {before - keep.sum():,} Gaussians on or under it; "
                  f"its normal is {tilt:.1f} deg from the camera estimate of up, and is used as up")
            up = n
    return keep, up


# Voxelizing

def voxelize(points, scales, quats, alpha, rgb, voxel, rng):
    """
    Spread each Gaussian's opacity and color over the voxels it covers. Small Gaussians land in one
    voxel; larger ones are sampled at several points inside their ellipsoid so their whole extent counts.
    Returns grid origin, per-voxel summed opacity, and opacity-weighted mean color.
    """
    big = scales.max(1) * 2 / voxel                                 # extent (2 sigma) in voxels
    n = np.clip(np.ceil(big ** 2), 1, 32).astype(int)                # surfaces are 2-D, so ~area samples
    idx = np.repeat(np.arange(len(points)), n)
    w, x, y, z = quats[idx].T                                        # rotation matrix per sample's Gaussian
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    offset = np.clip(rng.standard_normal((len(idx), 3)), -2, 2) * scales[idx]
    offset[n[idx] == 1] = 0                                          # one sample: use the center
    samples = points[idx] + np.einsum("nij,nj->ni", R, offset)
    weight = alpha[idx] / n[idx]                                     # each Gaussian contributes its opacity once

    origin = samples.min(0)
    cell = np.floor((samples - origin) / voxel).astype(int)
    dims = cell.max(0) + 1
    flat = np.ravel_multi_index(cell.T, dims)
    density = np.bincount(flat, weights=weight, minlength=dims.prod()).reshape(dims)
    color = np.stack([np.bincount(flat, weights=weight * rgb[idx, c], minlength=dims.prod()) for c in range(3)], -1)
    color = color.reshape(*dims, 3) / np.maximum(density, 1e-9)[..., None]
    return origin, density, color


def match_blocks(colors):
    """Nearest block by CIELAB distance, which tracks perceived color difference far better than RGB distance."""
    names = list(BLOCKS)
    to_lab = lambda c: cv2.cvtColor(np.asarray(c, np.float32).reshape(-1, 1, 3) / 255, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    palette = to_lab([BLOCKS[n] for n in names])
    lab = to_lab(colors * 255)
    nearest = np.argmin(((lab[:, None] - palette[None]) ** 2).sum(-1), axis=1)
    return [names[i] for i in nearest]


# Writing the .schem (Sponge Schematic v2: gzipped NBT, read by WorldEdit / FAWE)

def _name(s):
    b = s.encode()
    return struct.pack(">H", len(b)) + b


def nbt_int(name, v): return b"\x03" + _name(name) + struct.pack(">i", v)
def nbt_short(name, v): return b"\x02" + _name(name) + struct.pack(">h", v)
def nbt_bytes(name, b): return b"\x07" + _name(name) + struct.pack(">i", len(b)) + b
def nbt_ints(name, vals): return b"\x0b" + _name(name) + struct.pack(f">i{len(vals)}i", len(vals), *vals)
def nbt_compound(name, items): return b"\x0a" + _name(name) + b"".join(items) + b"\x00"
def nbt_empty_list(name): return b"\x09" + _name(name) + b"\x0a" + struct.pack(">i", 0)


def varints(values):
    """Each palette index as a VarInt: 7 bits per byte, high bit set while more bytes follow."""
    out = bytearray()
    for v in values:
        while v >= 0x80:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        out.append(v)
    return bytes(out)


def write_schem(path, blocks, data_version):
    """blocks: [x, y, z] array of block names (None = air)."""
    W, H, L = blocks.shape
    names = ["minecraft:air"] + sorted({f"minecraft:{b}" for b in blocks.flat if b})
    index = {n: i for i, n in enumerate(names)}
    order = blocks.transpose(1, 2, 0).ravel()  # the format's order: x fastest, then z, then y
    data = varints(index["minecraft:air" if b is None else f"minecraft:{b}"] for b in order)
    root = nbt_compound("Schematic", [
        nbt_int("Version", 2), nbt_int("DataVersion", data_version),
        nbt_short("Width", W), nbt_short("Height", H), nbt_short("Length", L),
        nbt_ints("Offset", [0, 0, 0]),
        # Paste centered on the player, standing on the build's bottom layer.
        nbt_compound("Metadata", [nbt_int("WEOffsetX", -(W // 2)), nbt_int("WEOffsetY", 0), nbt_int("WEOffsetZ", -(L // 2))]),
        nbt_int("PaletteMax", len(names)),
        nbt_compound("Palette", [nbt_int(n, i) for n, i in index.items()]),
        nbt_bytes("BlockData", data),
        nbt_empty_list("BlockEntities"),
    ])
    with gzip.open(path, "wb") as f:
        f.write(root)
    return names


def save_preview(path, blocks):
    """The build seen from front, right, back, left and top, one pixel per block, scaled up."""
    rgb = np.zeros(blocks.shape + (3,), np.uint8)
    solid = blocks != None  # noqa: E711 (elementwise comparison on an object array)
    for name in set(blocks[solid]):
        rgb[blocks == name] = BLOCKS[name]

    def project(occ, col, axis):
        """First solid block along `axis`, darkened with depth so shapes read."""
        hit = occ.any(axis)
        first = np.argmax(occ, axis)
        img = np.take_along_axis(col, np.expand_dims(first, axis)[..., None], axis).squeeze(axis)
        shade = 1 - 0.5 * first / max(occ.shape[axis], 1)
        img = (img * shade[..., None]).astype(np.uint8)
        img[~hit] = (245, 245, 245)
        return img

    views = [project(solid, rgb, 2).transpose(1, 0, 2)[::-1],                    # front (looking along +z)
             project(solid[::-1], rgb[::-1], 0).transpose(1, 0, 2)[::-1],        # right side
             project(solid[:, :, ::-1], rgb[:, :, ::-1], 2).transpose(1, 0, 2)[::-1, ::-1],  # back
             project(solid, rgb, 0).transpose(1, 0, 2)[::-1, ::-1],              # left side
             project(solid[:, ::-1], rgb[:, ::-1], 1).transpose(1, 0, 2)]        # top
    s = max(1, 256 // max(blocks.shape))
    tiles = [cv2.resize(v, (v.shape[1] * s, v.shape[0] * s), interpolation=cv2.INTER_NEAREST) for v in views]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(path), np.hstack(tiles)[..., ::-1])


# Main

def convert(args):
    rng = np.random.default_rng(0)
    means, scales, alpha, rgb, quats = read_ply(args.ply)
    print(f"{len(means):,} Gaussians in {args.ply}")
    cams = load_cameras(args.cameras) if args.cameras else None
    keep, up = find_object(means, alpha, cams, args, rng)

    # Y-up coordinates; the object's long horizontal axis along X, so long edges don't staircase.
    R = rotation_to_y(up)
    pts = means @ R.T
    flat = pts[keep][:, [0, 2]] - pts[keep][:, [0, 2]].mean(0)
    _, _, vt = np.linalg.svd(flat * alpha[keep, None], full_matrices=False)
    yaw = np.arctan2(vt[0, 1], vt[0, 0])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    R = Ry @ R
    pts = means @ R.T
    quats = (Rotation.from_matrix(R) * Rotation.from_quat(quats, scalar_first=True)).as_quat(scalar_first=True)

    # Voxel size from the object's robust extent (1-99th percentile ignores stray floaters).
    lo, hi = np.percentile(pts[keep], 1, axis=0), np.percentile(pts[keep], 99, axis=0)
    voxel = (hi - lo).max() / args.size
    inside = keep & np.all((pts >= lo - 2 * voxel) & (pts <= hi + 2 * voxel), axis=1)
    inside &= scales.max(1) < 3 * voxel  # blobs bigger than a few blocks are haze, not surface
    origin, density, color = voxelize(pts[inside], scales[inside], quats[inside], alpha[inside], rgb[inside], voxel, rng)

    solid = density >= args.density
    labels, n = ndimage.label(solid, structure=np.ones((3, 3, 3)))
    if n > 1:  # keep the biggest connected piece; floaters and leftovers are separate pieces
        sizes = ndimage.sum(solid, labels, range(1, n + 1))
        solid = labels == (np.argmax(sizes) + 1)
        print(f"Kept the largest of {n} connected pieces ({int(sizes.max()):,} of {int(sizes.sum()):,} voxels)")
    measured = solid.copy()
    if args.close:  # grow by a block then shrink back: seals pinholes without changing the outline
        k = args.close
        padded = np.pad(solid, k)  # room at the edges, or the shrink step eats the border
        solid |= ndimage.binary_closing(padded, np.ones((3, 3, 3)), iterations=k)[k:-k, k:-k, k:-k]
    if args.fill:
        solid = ndimage.binary_fill_holes(solid)
    # Colors: average each block with its neighbors (weighted by opacity) to calm per-voxel speckle,
    # then give every added block (sealed gaps, filled inside) the color of the nearest measured one.
    if not args.no_smooth:
        w = density * measured
        num = np.stack([ndimage.uniform_filter(color[..., c] * w, 3) for c in range(3)], -1)
        color = num / np.maximum(ndimage.uniform_filter(w, 3), 1e-9)[..., None]
    _, near = ndimage.distance_transform_edt(~measured, return_indices=True)
    color = color[tuple(near)]

    # Trim to the occupied box
    occupied = np.argwhere(solid)
    a, b = occupied.min(0), occupied.max(0) + 1
    solid, color = solid[a[0]:b[0], a[1]:b[1], a[2]:b[2]], color[a[0]:b[0], a[1]:b[1], a[2]:b[2]]

    blocks = np.full(solid.shape, None, dtype=object)
    blocks[solid] = match_blocks(color[solid])
    names = write_schem(args.out, blocks, args.data_version)
    W, H, L = solid.shape
    print(f"{int(solid.sum()):,} blocks, {W} x {H} x {L} (width x height x length), {len(names) - 1} block types "
          f"-> {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} KB)")
    preview = Path(args.preview) if args.preview else Path(args.out).with_suffix(".png")
    save_preview(preview, blocks)
    print(f"Preview: {preview}")
    return blocks


def main():
    convert(parse_args())


if __name__ == "__main__":
    main()
