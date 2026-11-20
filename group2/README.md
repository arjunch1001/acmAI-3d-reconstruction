# Group 2: frames → camera poses → Gaussian Splat → Minecraft

Only training needs a GPU, so the pipeline is split: the laptop does everything else.

```
laptop:  frames ─► check_images.py ─► run_sfm.py ─► dataset.zip           ~2 min
Colab:   dataset.zip ─► train_colab.ipynb ─► splat.ply                      ~10 min
laptop:  splat.ply ─► voxel/ply_to_schem.py ─► object.schem                 ~10 s
```

## 1. Laptop: frames to cameras

```bash
pip install -r group2/requirements.txt -r voxel/requirements.txt
python group2/check_images.py FRAMES --resize-to FRAMES_1280
python group2/run_sfm.py FRAMES_1280 WORK
python group2/sfm_report.py WORK FRAMES_1280
```

`check_images.py` stops on unreadable or mixed-size images. `sfm_report.py` must end with **OK to train**;
it also writes `WORK/sfm_report.png` (the camera ring). `run_sfm.py` writes `WORK/dataset.zip`.

## 2. Colab: train

Open `group2/train_colab.ipynb` in Colab (File → Open notebook → GitHub), pick a **T4 GPU** runtime,
run the cells in order, upload `WORK/dataset.zip`, and download `outputs.zip` (contains `splat.ply`).

Training runs in its own Python 3.10 environment built from `colab_requirements.txt`. Every package there
is prebuilt (gsplat ships compiled GPU code for torch 2.4 / CUDA 12.4), so setup takes a few minutes and
never compiles. Don't install gsplat into Colab's own Python: it has no prebuilt package for it and
compiling it there can run the machine out of memory.

## 3. Laptop: Minecraft

```bash
python voxel/ply_to_schem.py splat.ply object.schem --cameras WORK/dataset/sparse/0 --size 64
```

Writes `object.schem` and a preview `object.png`. In Minecraft Java with WorldEdit: put the `.schem` in
WorldEdit's `schematics` folder, then `//schem load object` and `//paste`.
