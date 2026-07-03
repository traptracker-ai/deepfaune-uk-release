# DeepFaune-UK Pipeline

A reproducible, staged pipeline for building a UK camera-trap species classifier
and continually improving it as new images arrive. You bring your own detector,
classifier, and images; the pipeline handles the ingest → verify → merge →
retrain → evaluate loop.

---

## What this is

A two-stage **detect-then-classify** camera-trap pipeline with a human-in-the-loop
retraining workflow:

- A **YOLO detector** (yours) locates animals.
- A **DeepFaune-UK classifier** (a fine-tuned DINOv2 ViT-L) names each crop.
- New images are auto-labelled, **verified by a human in Label Studio**, merged
  into an append-only pool, and the classifier is retrained from scratch on a
  versioned manifest.

**Core principle:** the dataset is the durable asset; models are reproducible
rebuilds from versioned manifests. You never overwrite old data — you only add
to it, and every model version can be traced to the exact data that made it.

---

## What you need to provide

This release contains the **pipeline only** — no models, no data. You supply:

1. **A YOLO detector in ONNX format.** Export from your `.pt` with:
   ```python
   from ultralytics import YOLO
   YOLO('your_detector.pt').export(format='onnx', imgsz=640, opset=12)
   ```
   Put the `.onnx` in `work/`.

2. **A classifier in ONNX format** with class metadata embedded (the pipeline
   reads `classes`, `input_size`, `mean`, `std` from the ONNX metadata). If you
   don't have one yet, you can train one — see "Cold start" below. Put it in
   `dataset/models/v1.0/`.

3. **An existing YOLO-format dataset** (`train/val/test` with `images/` and
   `labels/` subfolders) to seed the pool. Mounted read-only, used once.

4. **New camera-trap images** to grow the dataset over time.

> The class scheme is defined in `lib/dataset_lib.py` (the `CLASSES` list).
> Edit it to match your species before you start.

---

## Requirements

- **Docker Desktop** with the **NVIDIA Container Toolkit** (GPU passthrough).
  Training a ViT-L on CPU is not practical.
- An **NVIDIA GPU** (tested on an RTX A6000, 48 GB). ~8–12 GB is enough for the
  linear-probe workflow.
- Confirm GPU passthrough works before starting:
  ```
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

---

## Part 0 — Setup

### 0.1 Configure your class list
Open `lib/dataset_lib.py` and edit the `CLASSES` list to your species, in a
fixed order. This order is authoritative — it defines the class IDs used
everywhere. Do the same for `YOLO_NAMES` in `lib/inference_lib.py` (your
detector's class map, which may include Person/vehicle classes).

### 0.2 Point the compose file at your data
Edit the one marked line in `docker-compose.yml`:
```yaml
      - "C:/path/to/your/YOLO-Dataset:/existing:ro"
```
Change the left side to your existing dataset's location. (On macOS/Linux use a
normal path, e.g. `/home/you/data/YOLO-Dataset:/existing:ro`.)

### 0.3 Place your models
- Detector ONNX → `work/your_detector.onnx`
- Classifier ONNX → `dataset/models/v1.0/your_classifier.onnx`

### 0.4 Start the services
```
docker compose up --build
```
First build pulls PyTorch + Label Studio (several GB, once). Then:
- **TrapTracker dashboard** → http://localhost:5000  (start here — links to everything)
- **JupyterLab** → http://localhost:8888/lab  (no login)
- **Label Studio** → http://localhost:8080  (create a local account first time)

The **TrapTracker dashboard** is a local branded landing page: it shows live
status (pool size, current dataset/model version, latest test F1, whether a batch
is awaiting verification) and links to JupyterLab and Label Studio. It reads the
dataset folders read-only — it doesn't run anything itself. It's the easiest way
to see where the pipeline stands at a glance.

### 0.5 Add your logo (optional)
Drop a `logo.png` (or `.svg`/`.jpg`) into `dashboard/static/` and it appears in
the dashboard header automatically. Without one, the built-in TRAP TRACKER
wordmark is shown. See `dashboard/static/README.txt`.

### 0.6 Verify the environment
Open `work/00_setup.ipynb`, run all cells. Confirm:
- `GPU True` and your card is named,
- all `/dataset/...` folders report `OK`,
- your detector and classifier ONNX files are found.

**GPU check that matters:** the setup notebook confirms the classifier ONNX
loads on `CUDAExecutionProvider`, not CPU. If it falls back to CPU, the GPU
library path in the compose file needs attention (it's pre-configured for the
standard PyTorch image; only edit if your base image differs).

---

## Part 1 — Seed the pool (one-time)

The pool starts empty. Import your existing dataset as version `v1.0`.

Run `work/00b_bootstrap.ipynb`. It copies each labelled image into the pool
under a **content-hash filename** (SHA-256 based) and writes the `v1.0`
manifest. The content hash gives you automatic de-duplication: the same image
added twice is stored once.

After this you have a reproducible baseline: `v1.0 = these images, this split`.

> Splits are assigned deterministically from each image's content hash, so an
> image always lands in the same train/val/test split across every rebuild —
> no leakage.

---

## Part 2 — The iteration loop

This is the repeatable workflow each time you get new images.

### Step 1 — Drop in new images
Copy new **unlabelled** camera-trap images into `dataset/incoming/`.

### Step 2 — Auto-label  (`01_ingest_autolabel.ipynb`)
Runs your detector + classifier over the incoming images and writes Label Studio
pre-annotations. It reports how many detections are high- vs low-confidence, so
you know where your review attention matters. Produces
`dataset/ls_import/tasks.json` and `label_config.xml`.

### Step 3 — Verify in Label Studio  (`02_VERIFY_IN_LABEL_STUDIO.md`)
The only step in the browser. At http://localhost:8080:
1. Create a project.
2. **Settings → Labeling Interface**: paste `dataset/ls_import/label_config.xml`.
3. **Settings → Cloud Storage → Add Source → Local files**:
   - Absolute local path: `/label-studio/files`
   - tick **"Treat every bucket object as a source file"**, then **Sync**.
   - *(This is the step people get wrong. If images show broken, the storage
     path or sync is the cause.)*
4. **Import** `dataset/ls_import/tasks.json` — boxes + species appear as
   pre-annotations.
5. Review each task: approve correct ones, fix boxes/labels on the rest, delete
   false boxes. Prioritise the low-confidence detections.
6. **Export → JSON** to `dataset/ls_export/export.json`.

> **What gets added to the pool:** only images you *submit* in Label Studio.
> Tasks you never open are ignored. By default, images whose verified label is
> empty (blank frames) are **skipped** (the classifier trains on crops, so empty
> frames add nothing). Edit `03_merge.ipynb` if you want to keep them.

### Step 4 — Merge  (`03_merge.ipynb`)
Set `NEW_VERSION` (e.g. `'v1.1'`), run. Converts verified labels to YOLO format,
appends to the append-only pool (idempotent — re-running won't duplicate), and
writes a new versioned manifest.

### Step 5 — Retrain  (`04_retrain.ipynb`)
Run all cells. It:
1. **Materialises crops to disk once** (`dataset/crops/<version>/`) — this makes
   training fast by pre-cutting every crop so epochs read small files instead of
   decoding full frames repeatedly. It shows progress and runs in parallel.
2. Trains from those cached crops (linear probe from the DeepFaune backbone).

Defaults to **probe-only** (`P2_EPOCHS=0`) — the frozen-backbone probe typically
reaches deployment-grade accuracy in 1–2 epochs. Set `P2_EPOCHS` > 0 for a full
fine-tune if you need it. Best weights save to `dataset/models/<version>/best.pt`.

> **Speed note:** the crop-materialisation step is a one-time cost per version.
> Re-running training on the same version skips it (crops already exist). If
> training is slow, check `nvidia-smi` — 99% GPU-util means it's compute-bound
> and working; sawtoothing util means I/O-bound (rare with cached crops).

### Step 6 — Evaluate & export  (`05_evaluate.ipynb`)
Self-contained (rebuilds the model + test loader). Evaluates on the held-out
test split, prints the per-class report and macro-F1, exports a versioned ONNX
with class metadata, and writes a model card. Compare the macro-F1 against your
previous version to confirm the new data helped.

If the new model is as good or better, point Step 2's classifier path at it for
the next loop.

### Step 6 — Inference on new / OOD images  (`06_inference.ipynb`)
Run the **latest** model over a folder of unlabelled images (set `OOD_DIR` in the
notebook, e.g. `/dataset/ood_images`). Outputs annotated images (green=animal,
red=person, grey=vehicle) plus `predictions.csv` (every detection with species +
confidence + box) and `person_detections.csv` (people only, for security review).
No ground truth needed — this is batch prediction/deployment. On genuinely
out-of-distribution imagery, lower `DET_CONF` if the detector misses animals.

### Step 7 — Cleanup  (`07_cleanup.ipynb`)
Prepares for the next iteration by removing disposable temporary state (cached
crops, Label Studio hand-off files, and — opt-in — merged incoming images).
**Never touches** the pool, manifests, or models.

Safety: it only reports by default; set `CONFIRM=True` to delete. Clearing
`incoming/` is a separate opt-in and verifies every image is already in the pool
before removing it.

---

## Cold start (no classifier yet)

If you don't have a DeepFaune-UK classifier ONNX to begin with:
1. Seed the pool from your labelled dataset (Part 1).
2. Run `04_retrain.ipynb` directly on the `v1.0` manifest — it trains a
   classifier from the DeepFaune backbone on your existing data.
3. Run `05_evaluate.ipynb` to export the ONNX.
4. You now have a `v1.0` classifier; proceed to the iteration loop.

You still need the DeepFaune backbone checkpoint
(`deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt`) in `work/` — download it
from the DeepFaune project. And your YOLO detector ONNX in `work/`.

---

## Directory layout

```
docker-compose.yml     two services: pipeline (JupyterLab+GPU), labelstudio
Dockerfile             pipeline image
requirements.txt       pinned deps (GPU-ready, no numpy conflicts)
lib/
  dataset_lib.py       pool, hash-based splits, versioned manifests
  inference_lib.py     two-stage ONNX detect+classify (no Ultralytics runtime)
  labelstudio_lib.py   pre-annotation format + verified-label parsing
dashboard/
  app.py                   TrapTracker status dashboard (Flask, local only)
  templates/index.html     branded landing page
work/
  00_setup.ipynb           environment + GPU check
  00b_bootstrap.ipynb      seed pool from existing dataset (once)
  01_ingest_autolabel.ipynb
  02_VERIFY_IN_LABEL_STUDIO.md
  03_merge.ipynb
  04_retrain.ipynb         cached-crop training (fast)
  05_evaluate.ipynb        self-contained eval + ONNX export
  06_inference.ipynb       batch predict on new/OOD images (images + CSV)
  07_cleanup.ipynb         safe reset between iterations
dataset/
  incoming/    drop new unlabelled images here
  pool/        append-only store (images by content hash + YOLO labels)
  manifests/   dataset-vX.Y.json — reproducible version snapshots
  models/      one folder per model version (onnx + card + eval)
  (crops/, ls_import/, ls_export/ are created as needed)
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Classifier runs on CPU (slow), `libcublasLt`/`libnvrtc` warning | GPU library path — the compose file sets `LD_LIBRARY_PATH`; recreate the container with `docker compose down && up`. |
| `0 images to process` | images aren't directly in `dataset/incoming/`, or wrong extension. |
| Broken images in Label Studio | local storage not synced / wrong root path (Step 3.3). |
| `/dataset` paths `MISS` in Stage 0 | volume mount wrong in `docker-compose.yml`. |
| `/existing` not found in bootstrap | add/fix the `:/existing:ro` mount line, then `down && up`. |
| Training slow, GPU-util sawtooths | I/O-bound; ensure the crop cache built (Step 5) — cached crops fix this. |
| `pos_embed` size mismatch in training | `IMG_SIZE` must be 518 to match the DeepFaune backbone. |
| `broken data stream` reading an image | a corrupt file; the notebooks skip it and continue (`LOAD_TRUNCATED_IMAGES`). |
| Reproduce an old model | replay its manifest: set `MANIFEST_VERSION` in Stage 4. |

---

## The mental model

```
incoming/ (new, unlabelled)
      │  Stage 1: auto-label
      ▼
Label Studio (you verify)         ← the only human step
      │  Stage 3: merge
      ▼
pool/ (append-only) ──► manifest vX.Y   ← reproducible dataset version
      │  Stage 4: retrain from manifest (cached crops, seeded)
      ▼
models/vX.Y/ (onnx + card + eval)   ← versioned, comparable
      │
      └──► becomes the model Stage 1 uses next loop
```

Every arrow is a notebook except the Label Studio step. Every version is
reproducible. You never lose old data or old models.

---

## Attribution & licensing

This pipeline fine-tunes from the **DeepFaune** camera-trap backbone. If you
release a model built with it, follow DeepFaune's licence terms and cite:

> Rigoudy, N., et al. (2023). The DeepFaune initiative: a collaborative approach
> to wildlife recognition in camera-trap images.

Confirm the current DeepFaune licence (CC BY-SA / BY-NC-SA) before any
commercial use; ShareAlike and DeepFaune attribution are required either way.
