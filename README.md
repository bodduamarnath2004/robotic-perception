# CP260-2026 Final Project — Metric-Semantic 3D Scene Reconstruction

> **Desktop Socket Pose Estimation via Multi-View YOLO Detection and Ray Triangulation**

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Code](#running-the-code)
- [Output Format](#output-format)
- [Results](#results)
- [References](#references)

---

## Overview

This project implements a **Metric-Semantic 3D pose estimation pipeline** for desktop IO-panel sockets. Given a set of posed RGB images of a desktop scene, the system:

1. **Manually annotated** 16 frames of the desktop scene using **LabelMe**, labelling three socket classes: Power Socket, Ethernet Socket, and VGA Socket (bonus).
2. **Trained a YOLOv9m model** on these annotations to detect sockets reliably across all 704 frames.
3. Lifts detections to **metric 3D space** using multi-view ray triangulation with known camera poses.
4. Estimates a full **Oriented Bounding Box (OBB)** — centre, extent, and rotation — for each socket.
5. Outputs results in the required `final_answers.json` format.

> **Key Contribution:** The training data (LabelMe polygon annotations) was created manually for this project — no pre-labelled socket dataset was used.

---

## Project Structure

```
.
├── src/
│   └── notebook3879446005.ipynb   # Main pipeline notebook (all stages)
├── docs/
│   └── CP260_Project_Report.pdf   # Project report
├── Dataset
├── README.md
```

### Notebook Cell Map

| Cell | Stage | Description |
|------|-------|-------------|
| 1 | Dataset Prep | Convert LabelMe JSON annotations → YOLO format, train/val split |
| 2 | YAML Config | Generate `data.yaml` for YOLO training |
| 3 | Install | `pip install ultralytics` |
| 4 | Training | Fine-tune YOLOv9m for 200 epochs with heavy augmentation |
| 5 | Detection | Run trained model on all 704 frames |
| 6 | Visualisation | Draw bounding boxes on detected frames |
| A | 3D — Poses | Load `poses.json` and camera intrinsics |
| B | 3D — Format | Adapt YOLO detections to 3D pipeline format |
| C | 3D — Triangulation | Multi-view ray intersection → 3D centres |
| D | 3D — OBB Helpers | Panel normal estimation, extent recovery |
| E | 3D — OBB Assembly | Build final OBBs, validate VGA vs sample answer |
| F | 3D — Save | Write and verify `final_answers.json` |
| G | 3D — Visualise | Project OBBs back onto image frames |

---

## System Architecture

```
Posed Images + poses.json
        │
        ▼
┌───────────────────┐
│  YOLO Dataset     │  LabelMe JSON → YOLO bounding boxes
│  Preparation      │  80/20 train/val split
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  YOLOv9m          │  Fine-tuned on 16 annotated frames
│  Training         │  200 epochs, AdamW, heavy augmentation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Multi-Frame      │  All 704 frames, conf threshold τ = 0.25
│  Detection        │  3 classes: Ethernet, Power, VGA
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Multi-View Ray   │  Pixel centre → ray in world space
│  Triangulation    │  Least-squares intersection of N rays
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  OBB Assembly     │  Panel normal → rotation matrix
│                   │  Reprojection → metric extent
└────────┬──────────┘
         │
         ▼
   final_answers.json
```

---

## Dataset

Download from:
Dataset folder

After downloading, place the contents so the structure looks like:

```
/Dataset
├── frame_000001.png
├── frame_000002.png
├── ...                  # 704 total frames
├── frame_000001.json    # LabelMe annotations (16 frames)
├── ...
└── poses.json           # Camera-to-world poses, keyed by frame index
```

### Manual Annotation

16 frames were manually annotated using **LabelMe** with polygon labels for each socket instance. Annotations were saved as `frame_*.json` files alongside the images. These are the only labelled frames in the dataset — all other 688 frames are unannotated and used only for detection inference.

**Annotation classes:**

| Class | LabelMe Label | YOLO ID |
|-------|--------------|--------|
| Ethernet Socket | `Ethernet Socket` | 0 |
| Power Socket | `Power Socket` | 1 |
| VGA Socket | `VGA Socket` | 2 |

### YOLO Training

YOLOv9m was trained on the 16 manually annotated frames for 200 epochs using the Ultralytics framework. Transfer learning from ImageNet-pretrained weights was used, with the backbone frozen for the first 10 layers to prevent overfitting on the small dataset. Heavy augmentation (Mosaic, MixUp, Copy-Paste, flips, rotations) was applied to artificially expand the effective training set.

### Camera Intrinsics

| Parameter | Value |
|-----------|-------|
| fx | 1477.00974684544 |
| fy | 1480.4424455584467 |
| cx | 1298.2501500778505 |
| cy | 686.8201623541711 |
| Width | 2560 px |
| Height | 1440 px |

---

## Requirements

```
python >= 3.9
torch >= 2.0
ultralytics
opencv-python
numpy
matplotlib
scikit-learn
Pillow
```

All dependencies are available on Kaggle by default except `ultralytics`, which is installed in Cell 3.

---

## Installation

### On Kaggle (recommended)

1. Upload the dataset to Kaggle as a dataset named `datasetrp` under your account.
2. Open the notebook (`src/notebook3879446005.ipynb`) in a Kaggle notebook with **GPU T4 x2** enabled.
3. Run all cells top to bottom.

### Locally

```bash
git clone <your-repo-url>
cd <repo>
pip install ultralytics opencv-python numpy matplotlib scikit-learn Pillow
```

Then update the following path constants in the notebook to point to your local dataset:

```python
SOURCE_DIR   = "/path/to/your/dataset"
DATASET_DIR  = "/path/to/your/dataset"
OUTPUT_DIR   = "./yolo_ethernet_dataset"
```

---

## Running the Code

Run cells **in order** from top to bottom. The pipeline is sequential — each stage depends on outputs from the previous one.

### Key outputs produced

| File | Description |
|------|-------------|
| `/kaggle/working/yolo_ethernet_dataset/` | YOLO-formatted dataset |
| `socket_training/yolov8n_multiclass_run1/weights/best.pt` | Trained model weights |
| `/kaggle/working/visualized_detections/` | Detection visualisations |
| `/kaggle/working/final_answers.json` | Final OBB predictions |
| `/kaggle/working/obb_projection.png` | OBB projected onto best frame |
| `/kaggle/working/obb_all_frames.png` | OBB projected onto all 16 frames |

### Fix model path (important)

If Cell 5 fails to load the model, replace the hardcoded path with:

```python
model_path = str(results.save_dir / "weights" / "best.pt")
```

---

## Output Format

`final_answers.json` contains a list of 3 entries:

```json
[
  {
    "entity": "vga_socket",
    "obb": {
      "center"  : [x, y, z],
      "extent"  : [width, height, depth],
      "rotation": [[r00, r01, r02],
                   [r10, r11, r12],
                   [r20, r21, r22]]
    }
  },
  {
    "entity": "ethernet_socket",
    "obb": { ... }
  },
  {
    "entity": "power_socket",
    "obb": { ... }
  }
]
```

All values are in **metres**. The rotation matrix is a valid element of SO(3) aligned with the back-panel normal.

---

## Results

### Detection (YOLOv9m, epoch 76/200)

| Metric | Value |
|--------|-------|
| Precision | 0.335 |
| Recall | 0.583 |
| mAP@50 | 0.316 |
| mAP@50:95 | 0.181 |

### VGA Socket Validation vs Sample Answer

| Metric | Value |
|--------|-------|
| Centre error | 0.25 |
| Rotation error | 177.8° |

---

## References

- Redmon et al., *You Only Look Once*, CVPR 2016
- Jocher et al., *Ultralytics YOLO*, 2023
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, Cambridge 2003
- McCormac et al., *SemanticFusion*, ICRA 2017
- Rosinol et al., *Kimera*, ICRA 2020
