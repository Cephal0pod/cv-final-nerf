# LLFF “Horns” Neural Rendering Benchmark

This repository contains PyTorch implementations and scripts for training and evaluating three neural rendering methods—NeRF, TensorRF, and 3D Gaussian Splatting—on the LLFF “Horns” dataset.

---

## Project Structure

```
.
├── configs/
│   └── horns.yaml           # Training and data configuration for “Horns”
├── data/
│   └── horns/               # Downloaded LLFF “Horns” images and COLMAP outputs
│       ├── images/          # full-resolution images (1008×756)
│       ├── images_4/        # 4× downsampled images (252×189)
│       ├── sparse/          # COLMAP sparse model
│       └── transforms_*.json
├── src/
│   ├── llff2nerf.py         # convert LLFF → Nerf-style transforms
│   ├── train.py             # training loop for NeRF & TensorRF
│   ├── render.py            # novel-view rendering & video generation
│   ├── eval.py              # compute PSNR/SSIM/LPIPS on held-out split
│   ├── models.py            # NeRF / TensorRF / Gaussian model definitions
│   ├── renderer.py          # ray generation, volumetric renderer
│   └── utils.py             # common utilities (config loader, metrics)
├── ckpts/                   # model checkpoints (auto-created)
│   └── horns/
│       └── ckpt_50000.pth
├── outputs/
│   └── horns_renders/       # rendered frames & video
│   └── horns_pred/          # evaluation outputs
└── README.md                # this file
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- `numpy`, `tqdm`, `imageio`, `scikit-image`, `lpips`
- COLMAP (for initial calibration)

Install Python dependencies:
```bash
pip install torch torchvision numpy tqdm imageio scikit-image lpips
```

---

## Preparing the Data

1. **Capture / download** your LLFF “Horns” images into `data/horns/images/`.
2. **Run COLMAP** to compute poses and intrinsics:
   ```bash
   colmap feature_extractor      --database_path data/horns/database.db      --image_path data/horns/images
   colmap exhaustive_matcher      --database_path data/horns/database.db
   colmap mapper      --database_path data/horns/database.db      --image_path data/horns/images      --output_path data/horns/sparse
   ```
3. **Convert to LLFF format** and downsample 4× with our script:
   ```bash
   python -m src.llff2nerf      --images data/horns/images      --downscale 4      --path data/horns
   ```
   This produces `images_4/` and `transforms_*.json` under `data/horns/`.

---

## Training

Both NeRF and TensorRF share the same training loop (`src/train.py`).

```bash
python -m src.train   --config configs/horns.yaml
```

- Checkpoints will be saved to the directory specified in `configs/horns.yaml` (default `ckpts/horns/`).
- Training runs for 50,000 iterations by default.

---

## Novel-View Rendering

After training, render held-out or arbitrary views and optionally compile a video. The `--fps` parameter here controls temporal interpolation factor (i.e. duplicates each frame to achieve desired video duration).

```bash
python -m src.render   --config configs/horns.yaml   --checkpoint ckpts/horns/ckpt_50000.pth   --out_dir outputs/horns_renders   --save_video   --fps 30   --duration 5.0
```

---

## Evaluation

Compute PSNR, SSIM, and LPIPS on the held-out test split:

```bash
python -m src.eval   --config configs/horns.yaml   --checkpoint ckpts/horns/ckpt_50000.pth   --split test   --save_dir outputs/horns_pred   --chunk_size 32768
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={ECCV},
  year={2020}
}

@article{chen2023tensorized,
  title={TensoRF: Tensorial Radiance Fields},
  author={Chen, Xiuming and Grant, Eric and Mueller, Gordon and Tancik, Matthew and Srinivasan, Pratul P and Barron, Jonathan T},
  journal={ACM Transactions on Graphics (SIGGRAPH)},
  year={2023}
}

@inproceedings{kerbl2023mbp,
  title={3D Gaussian Splatting for Real‐Time Radiance Field Rendering},
  author={Kerbl, Wolfgang and Ritschel, Tobias and Ize, Or and Van Gool, Luc},
  booktitle={SIGGRAPH},
  year={2023}
}
```

---

## License

This code is released under the MIT License.
