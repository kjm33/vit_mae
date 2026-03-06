#!/bin/bash

# Ustawienie widoczności kart (jeśli masz ich więcej w systemie)
export CUDA_VISIBLE_DEVICES=0,1

# Uruchomienie przez accelerate launch
# --multi_gpu: wykorzystaj obie karty
# --num_processes 2: uruchom 2 instancje (po jednej na GPU)
# --mixed_precision bf16: włącz precyzję BF16 dla Ampere
accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision bf16 \
    train_mae.py