# Training MAE from Scratch on Your Images (2× 24GB GPUs)

## 1. Prepare your image dataset

The pretrain script uses **ImageFolder**: images must live under a `train` subfolder, with each **class** in its own subfolder. For pretraining you don't need real labels, so you can use a single “dummy” class.

**Recommended layout:**

```
/path/to/your/images/
  train/
    all/          ← one folder is enough for pretraining
      img001.jpg
      img002.png
      ...
```

Or, if you have real classes (e.g. for later fine-tuning):

```
/path/to/your/images/
  train/
    class_a/
      img1.jpg
    class_b/
      img2.png
```

Supported formats: whatever `PIL`/`torchvision` can read (e.g. jpg, png).

---

## 2. Choose model and image size

- **ViT-Base** (`mae_vit_base_patch16`): fits comfortably on 2× 24GB; good default.
- **ViT-Large** (`mae_vit_large_patch16`): possible with smaller batch and gradient accumulation.

Your `main_pretrain.py` is set up for **custom resolution** and **grayscale** by default (`input_height=32`, `input_width=512`, `in_chans=1`). For standard **RGB 224×224** images, override:

- `--input_height 224 --input_width 224 --in_chans 3`

---

## 3. Launch distributed training (2 GPUs)

Use **torchrun** (PyTorch ≥ 1.9) so that `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` are set automatically.

**Example: ViT-Base, RGB 224×224, 2 GPUs**

```bash
torchrun --nproc_per_node=2 main_pretrain.py \
  --data_path /path/to/your/images \
  --output_dir ./output_pretrain \
  --log_dir ./output_pretrain/logs \
  --model mae_vit_base_patch16 \
  --input_height 224 \
  --input_width 224 \
  --in_chans 3 \
  --batch_size 32 \
  --accum_iter 2 \
  --epochs 400 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --mask_ratio 0.75 \
  --norm_pix_loss \
  --num_workers 8
```

- **Effective batch size** = `batch_size × accum_iter × num_gpus` = 32 × 2 × 2 = **128**. Learning rate is scaled with this (base LR 256 rule).
- If you hit **OOM** on 24GB: reduce `--batch_size` (e.g. 16 or 8) and optionally increase `--accum_iter` to keep effective batch size (e.g. 16×4×2=128).

**Example: ViT-Large (tighter on 24GB)**

```bash
torchrun --nproc_per_node=2 main_pretrain.py \
  --data_path /path/to/your/images \
  --output_dir ./output_pretrain_large \
  --model mae_vit_large_patch16 \
  --input_height 224 --input_width 224 --in_chans 3 \
  --batch_size 8 \
  --accum_iter 8 \
  --epochs 400 \
  --warmup_epochs 40 \
  --blr 1.5e-4 --weight_decay 0.05 \
  --mask_ratio 0.75 --norm_pix_loss
```

Effective batch size = 8×8×2 = 128.

---

## 4. Optional: gradient accumulation only (single process, 1 GPU)

If you prefer one process and one GPU (no distributed):

```bash
python main_pretrain.py \
  --data_path /path/to/your/images \
  --output_dir ./output_pretrain \
  --model mae_vit_base_patch16 \
  --input_height 224 --input_width 224 --in_chans 3 \
  --batch_size 32 \
  --accum_iter 4 \
  --epochs 400 \
  --blr 1.5e-4 --weight_decay 0.05 \
  --mask_ratio 0.75 --norm_pix_loss
```

No `torchrun`; `world_size` stays 1. Effective batch = 32×4×1 = 128.

---

## 5. Resume from checkpoint

```bash
torchrun --nproc_per_node=2 main_pretrain.py \
  ... same args ... \
  --resume ./output_pretrain/checkpoint-20.pth
```

---

## 6. Hyperparameter summary

| Argument        | Role |
|----------------|------|
| `--data_path`  | Root directory containing `train/` (and optionally `val/`) with class subfolders. |
| `--batch_size` | Per-GPU batch size. |
| `--accum_iter` | Gradient accumulation steps; effective batch = batch_size × accum_iter × num_gpus. |
| `--blr`        | Base LR; actual LR = blr × effective_batch_size / 256. |
| `--mask_ratio` | Fraction of patches masked (e.g. 0.75). |
| `--norm_pix_loss` | Use normalized pixel targets (recommended). |
| `--epochs` / `--warmup_epochs` | Total and warmup epochs. |

Checkpoints are written to `--output_dir` every 20 epochs (and at the last epoch). TensorBoard logs go to `--log_dir`.
