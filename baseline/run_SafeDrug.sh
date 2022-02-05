#!/bin/bash
python run_baseline.py \
    --train \
    --eval \
    --data_prefix safedrug \
    --model_name SafeDrug \
    --learning_rate 5e-4 \
    --epochs 50 \
    --alpha_bce 0.95 \
    --alpha_margin 0.05 \
    --emb_dim 64 \
    --dropout 0.5 \
    --target_ddi 0.06 \
