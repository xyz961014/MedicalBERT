# MedicalBERT

MedicalBERT is a pre-training and fine-tuning based framework for fusing heterogeneous information in electronic health records (EHR) data for medication recommendation. We adopt the MIMIC-III dataset, please refer to  https://physionet.org/content/mimiciii/1.4/ to download the dataset.

The MedicalBERT framework is divided into three steps: 

### Data preprocessing

Get data prepared for pre-training:

```bash
cd data
mkdir processed_data

python preprocess_data.py --data_path PATH/TO/MIMIC-III --pretrain --labevents --static --save processed_data/pretrain

python create_vocab_and_records.py --data_dir PATH/TO/MIMIC-III --data_path processed_data/pretrain_data.pkl --pretrain --first_day --inner_temporal --save processed_data/pretrain

python create_pretrain_data.py --vocab_file processed_data/pretrain_vocab.pkl --input_id_file processed_data/pretrain_id_data.txt --static_prob 0 --lab_prob 0 --value_prob 0 --flag_prob 0 --dupe_factor 30 --save processed_data/pretrain
```

Get data prepared for fine-tuning:

```bash
python preprocess_data.py --data_path PATH/TO/MIMIC-III --multi_visit --save processed_data/finetune

python create_vocab_and_records.py --data_dir PATH/TO/MIMIC-III --data_path processed_data/finetune_data.pkl --save processed_data/finetune 

python find_lab_data_for_task.py --data_path processed_data/finetune_data.pkl --data_dir PATH/TO/MIMIC-III --save processed_data/finetune_lab

python find_static_data_for_task.py --data_path processed_data/finetune_data.pkl --static_data_path processed_data/pretrain_data.pkl --save processed_data/safedrug_lab_static
```

### Pre-training

Build model config:

``` bash
cd ../script
python build_model_config_from_vocab.py \
    --vocab_file ../data/processed_data/pretrain_vocab.pkl \
    --save ./model_config \
    --attention_probs_dropout_prob 0.1 \
    --hidden_act gelu \
    --hidden_dropout_prob 0.1 \
    --hidden_size 768 \
    --intermediate_size 3072 \
    --max_position_embeddings 512 \
    --num_attention_heads 16 \
    --num_hidden_layers 12 \
    --only_mlm \
```

Then pre-train:

```bash
python run_pretrain.py \
    --data_dir ../data/processed_data/pretrain_seq_len_512_max_pred_80_mlm_prob_0.15_random_seed_12345_dupe_30 \
    --config_file ./model_config.json \
    --vocab_file ../data/processed_data/pretrain_vocab.pkl \
    --output_dir . \
    --train_batch_size 2 \
    --max_steps 200000 \
    --seed 12345 \
    --gradient_accumulation_steps 2 \
    --num_steps_per_checkpoint 1000 \
    --log_freq 10 \
    --display_freq 100 \
    --devices 0 1 2 3 \
    --distributed \
```

### Fine-tuning

```bash
python run_finetune.py \
    --data_dir ../data/processed_data \
    --pretrained_model_path . \
    --vocab_file ../data/processed_data/pretrain_vocab.pkl \
    --output_dir medication_recommendation \
    --task_name medication_recommendation_safedrug_lab_static \
    --train_batch_size 1 \
    --max_epochs 5 \
    --seed 42 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_steps_per_checkpoint 1000 \
    --log_freq 20 \
    --display_freq 100 \
    --devices 0 \
    --alpha_bce 0.9 \
    --alpha_margin 0.02 \
    --attention_probs_dropout_prob 0.1 \
    --hidden_dropout_prob 0.1 \
    --weight_decay 0.0 \
    --mean_repr \
    --eval \
    --train \
    --decoder MLP \
    --decoder_mlp_layers 4 \
    --decoder_hidden 3072 \
    --no_lab \
```

