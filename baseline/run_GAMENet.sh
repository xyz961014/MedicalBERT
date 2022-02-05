#!/bin/bash
python run_baseline.py \
    --train \
    --eval \
    --data_prefix safedrug \
    --model_name GAMENet \
    --learning_rate 0.0002 \
    --epochs 40 \
    --emb_dim 64 \
    --dropout 0.4 \
