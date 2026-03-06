#!/usr/bin/env bash
# MAE pretrain from scratch on your images, 2 GPUs (24GB each)
# Edit DATA_PATH and OUTPUT_DIR, then run: bash run_pretrain_2gpu.sh

DATA_PATH="/path/to/your/images"   # must contain train/<class_folders>/
OUTPUT_DIR="./output_pretrain"

torchrun --nproc_per_node=2 main_pretrain.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$OUTPUT_DIR/logs" \
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
