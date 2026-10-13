"""
Smart Keyframe Picker for 3D Reconstruction.

Runs a pretrained YOLO model on a webcam or video, tracks one target object,
and automatically saves only the frames that are useful for 3D reconstruction:
the object is fully in view, the image is sharp, and the viewpoint is new.
"""

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 1. Command-line options
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Pick good keyframes of an object for 3D reconstruction.")
    p.add_argument("--source", default="0", help="webcam index (0) or path to a video file")
    p.add_argument("--target", default="cup", help="COCO class name to capture, e.g. cup, bottle, chair")
    p.add_argument("--model", default="yolo11n.pt", help="pretrained YOLO weights")
    p.add_argument("--conf", type=float, default=0.4, help="minimum detection confidence")
    p.add_argument("--margin", type=int, default=10, help="px the box must stay away from the frame edge")
    p.add_argument("--blur", type=float, default=0.8,
                   help="minimum sharpness as a fraction of this object's recent median sharpness")
    p.add_argument("--novelty", type=float, default=20.0, help="minimum difference from last saved view (0-255)")
    p.add_argument("--out", default="output", help="folder for keyframes, log, and summary")
    p.add_argument("--save-video", action="store_true", help="also save the annotated video")
    p.add_argument("--no-show", action="store_true", help="don't open a preview window")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 2. The three quality checks
# ---------------------------------------------------------------------------
def is_fully_visible(box, frame_w, frame_h, margin):
    """True if the box doesn't touch the frame edges (object isn't cut off)."""
    x1, y1, x2, y2 = box
    return x1 > margin and y1 > margin and x2 < frame_w - margin and y2 < frame_h - margin


def sharpness(crop):
    """Variance of the Laplacian: edges are strong in sharp images, weak in blurry ones."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def novelty(crop, last_crop):
    """Mean pixel difference between this view and the last saved view (0 = identical)."""
    if last_crop is None:
        return 255.0  # nothing saved yet, so any view is new
    a = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))
    b = cv2.resize(cv2.cvtColor(last_crop, cv2.COLOR_BGR2GRAY), (64, 64))
    return float(np.mean(cv2.absdiff(a, b)))


# ---------------------------------------------------------------------------
# 3. Choosing which detection to follow
# ---------------------------------------------------------------------------
def pick_target(boxes, locked_id):
    """
    Return (index, track_id) of the box to follow, or (None, locked_id) if nothing.
    Stay locked on the same track ID so every keyframe shows the same object;
    if that object disappears, switch to the most confident detection.
    """
    if boxes is None or len(boxes) == 0:
        return None, locked_id
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
    if locked_id in ids:
        return ids.index(locked_id), locked_id
    best = int(boxes.conf.argmax())
    return best, ids[best]


# ---------------------------------------------------------------------------
# 4. Drawing helpers
# ---------------------------------------------------------------------------
def draw_hud(frame, status, color, saved_count, target):
    """Status bar at the top of the frame."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, f"target: {target}   keyframes saved: {saved_count}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def make_summary(paths, out_file, thumb_w=240, cols=4):
    """Tile saved keyframes into one contact-sheet image."""
    if not paths:
        return
    thumbs = []
    for p in paths:
        img = cv2.imread(str(p))
        h = int(img.shape[0] * thumb_w / img.shape[1])
        thumbs.append(cv2.resize(img, (thumb_w, h)))
    h = thumbs[0].shape[0]
    thumbs = [cv2.resize(t, (thumb_w, h)) for t in thumbs]
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))  # pad last row with black tiles
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(str(out_file), np.vstack(rows))


# ---------------------------------------------------------------------------
# 5. Main loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.out)
    (out_dir / "keyframes").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    name_to_id = {name: i for i, name in model.names.items()}
    if args.target not in name_to_id:
        raise SystemExit(f"Unknown target '{args.target}'. Choose one of: {', '.join(model.names.values())}")
    target_id = name_to_id[args.target]

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    log_file = open(out_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["frame", "track_id", "conf", "sharpness", "novelty", "decision"])

    writer = None
    locked_id, last_crop, saved = None, None, []
    recent_sharp = deque(maxlen=30)  # last 30 sharpness scores, so "blurry" is relative to this object
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video (or camera disconnected)
        frame_idx += 1
        h, w = frame.shape[:2]

        # Detect + track only the target class. persist=True keeps IDs between frames.
        result = model.track(frame, persist=True, classes=[target_id], conf=args.conf, verbose=False)[0]
        annotated = result.plot()  # YOLO's own boxes + labels

        idx, locked_id = pick_target(result.boxes, locked_id)

        if idx is None:
            # No target in this frame: report it and keep going, never crash.
            status, color = f"no {args.target} detected", (0, 165, 255)
            log.writerow([frame_idx, "", "", "", "", "no_detection"])
        else:
            x1, y1, x2, y2 = result.boxes.xyxy[idx].int().tolist()
            conf = float(result.boxes.conf[idx])
            crop = frame[y1:y2, x1:x2]

            sharp = sharpness(crop) if crop.size else 0.0
            new = novelty(crop, last_crop) if crop.size else 0.0

            # "Sharp enough" = at least --blur x this object's typical (median) recent sharpness.
            # This adapts to the camera and to the object's texture (a smooth orange scores low
            # even when in focus).
            if crop.size:
                recent_sharp.append(sharp)
            min_sharp = args.blur * float(np.median(recent_sharp)) if recent_sharp else 0.0

            # Run the checks in order; the first one that fails is the reason we skip.
            if not is_fully_visible((x1, y1, x2, y2), w, h, args.margin):
                decision, color = "skip: cut off at edge", (0, 0, 255)
            elif len(recent_sharp) < 10:
                decision, color = "skip: learning sharpness", (255, 255, 0)
            elif sharp < min_sharp:
                decision, color = f"skip: blurry ({sharp:.0f} < {min_sharp:.0f})", (0, 0, 255)
            elif new < args.novelty:
                decision, color = f"skip: same view ({new:.0f} < {args.novelty:.0f})", (0, 255, 255)
            else:
                decision, color = "SAVED keyframe", (0, 255, 0)
                path = out_dir / "keyframes" / f"keyframe_{len(saved) + 1:03d}.jpg"
                cv2.imwrite(str(path), frame)  # save the clean frame, not the annotated one
                saved.append(path)
                last_crop = crop.copy()

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)  # highlight the followed object
            status = f"id {locked_id}  conf {conf:.2f}  sharp {sharp:.0f}  new {new:.0f}  -> {decision}"
            log.writerow([frame_idx, locked_id, f"{conf:.2f}", f"{sharp:.1f}", f"{new:.1f}", decision])

        draw_hud(annotated, status, color, len(saved), args.target)

        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)

        if not args.no_show:
            cv2.imshow("Keyframe Picker (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    log_file.close()
    make_summary(saved, out_dir / "summary.jpg")
    print(f"Processed {frame_idx} frames, saved {len(saved)} keyframes to {out_dir / 'keyframes'}")


if __name__ == "__main__":
    main()
