#!/bin/bash
python run_baseline.py \
    --train \
    --eval \
    --data_prefix safedrug \
    --model_name Retain \
    --learning_rate 0.0001 \
    --epochs 50 \
    --emb_dim 64 \
    --dropout 0.3 \
