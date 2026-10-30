import argparse
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import pycolmap

from check_images import list_images


# Command-line options

def parse_args():
    p = argparse.ArgumentParser(description="Estimate camera poses for an orbit image set with COLMAP (via pycolmap).")
    p.add_argument("images", help="folder of checked, resized frames (run check_images.py first)")
    p.add_argument("work", help="output folder: database, raw sparse models, and the undistorted dataset")
    p.add_argument("--matcher", choices=["auto", "circular", "exhaustive", "sequential"], default="auto",
                   help="which image pairs to match: circular = each frame with its next --window frames, "
                        "wrapping around so the end of the orbit is matched to the start; "
                        "auto = exhaustive on a CUDA build (fast there, slightly more accurate), else circular")
    p.add_argument("--window", type=int, default=10, help="neighbors per frame for circular/sequential matching")
    p.add_argument("--mapper", choices=["incremental", "global"], default="incremental",
                   help="incremental = classic COLMAP, global = GLOMAP")
    p.add_argument("--camera-model", default="OPENCV", help="COLMAP camera model shared by every frame")
    p.add_argument("--max-features", type=int, default=8192, help="SIFT keypoints kept per image")
    p.add_argument("--overwrite", action="store_true", help="delete the work folder first if it exists")
    return p.parse_args()


@contextmanager
def timed(label, times):
    """Print how long a stage took, so we know where the meeting's minutes go."""
    print(f"[{label}] running...", flush=True)
    start = time.perf_counter()
    yield
    times[label] = time.perf_counter() - start
    print(f"[{label}] done in {times[label]:.1f}s", flush=True)


# Matching

def make_circular_pairs(names, window, out_file):
    """
    Pair every frame with the next `window` frames, wrapping past the end.
    An orbit is a loop, so frame 100 sits right next to frame 1; sequential matching never
    compares them, which leaves a seam where drift is never corrected. Exhaustive matching
    compares everything (n^2 / 2 pairs) when only nearby views can actually overlap.
    """
    n = len(names)
    pairs = {tuple(sorted((names[i], names[(i + k) % n]))) for i in range(n) for k in range(1, window + 1)}
    out_file.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    return len(pairs)


def match(db, names, args, work):
    if args.matcher == "exhaustive":
        print(f"  {len(names) * (len(names) - 1) // 2} pairs")
        pycolmap.match_exhaustive(db)
    elif args.matcher == "sequential":
        opts = pycolmap.SequentialPairingOptions()
        opts.overlap = args.window
        pycolmap.match_sequential(db, pairing_options=opts)
    else:
        pairs_file = work / "pairs.txt"
        print(f"  {make_circular_pairs(names, args.window, pairs_file)} pairs")
        opts = pycolmap.ImportedPairingOptions()
        opts.match_list_path = pairs_file
        pycolmap.match_image_pairs(db, pairing_options=opts)


# Main

def main():
    args = parse_args()
    images, work = Path(args.images), Path(args.work)
    if work.exists():
        if not args.overwrite:
            raise SystemExit(f"{work} already exists (pass --overwrite to replace it)")
        shutil.rmtree(work)
    sparse, dataset = work / "sparse", work / "dataset"
    sparse.mkdir(parents=True)
    db = work / "database.db"

    names = [p.name for p in list_images(images)]
    if not names:
        raise SystemExit(f"No images found in {images}")
    if args.matcher == "auto":
        args.matcher = "exhaustive" if pycolmap.has_cuda else "circular"
    print(f"{len(names)} images, matcher={args.matcher}, mapper={args.mapper}, "
          f"camera={args.camera_model}, device={'cuda' if pycolmap.has_cuda else 'cpu'}")
    times = {}

    # 1. Keypoints + descriptors for each image. CameraMode.SINGLE: one phone, one lens, so
    #    all frames share intrinsics, which gives bundle adjustment far more data per parameter.
    with timed("extract", times):
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = args.camera_model
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.sift.max_num_features = args.max_features
        pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction)

    # 2. Descriptor matching + geometric verification for the chosen image pairs.
    with timed("match", times):
        match(db, names, args, work)

    # 3. Poses, intrinsics, and sparse 3D points. Writes sparse/0, sparse/1, ... if the
    #    images fall apart into disconnected groups.
    with timed("map", times):
        if args.mapper == "global":
            models = pycolmap.global_mapping(db, images, sparse)
        else:
            models = pycolmap.incremental_mapping(db, images, sparse)
    if not models:
        raise SystemExit("Mapping failed: no model could be built. Run sfm_report.py / check_images.py to see why.")
    best = max(models, key=lambda i: models[i].num_reg_images())

    # 4. Undistort into a pinhole dataset. Gaussian rasterizers assume an ideal pinhole camera,
    #    so lens distortion is removed from the pixels here, once, instead of in every trainer.
    with timed("undistort", times):
        pycolmap.undistort_images(dataset, sparse / str(best), images)
        # Inria 3DGS, gsplat and Nerfstudio's colmap parser all look for <dataset>/sparse/0.
        (dataset / "sparse" / "0").mkdir()
        for f in (dataset / "sparse").glob("*.bin"):
            f.rename(dataset / "sparse" / "0" / f.name)

    rec = models[best]
    print(f"\nModels: {len(models)}   largest (sparse/{best}): {rec.num_reg_images()}/{len(names)} images registered, "
          f"{rec.num_points3D()} points, reprojection error {rec.compute_mean_reprojection_error():.2f}px")
    print("Time: " + ", ".join(f"{k} {v:.0f}s" for k, v in times.items()) + f", total {sum(times.values()):.0f}s")
    print(f"Training dataset: {dataset}   (next: python group2/sfm_report.py {work} {images})")


if __name__ == "__main__":
    main()
