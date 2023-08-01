#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=generalization_experiment

# Prepare data
venv2/bin/python ${BASE_DIR}/prepare_base_data.py

# Preprocess
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/karpathy/train_data.json --output_file coco_stair_train_data_karpathy

# Train
venv2/bin/python train.py --data ./data/coco/coco_stair_train_data_karpathy.pkl --out_dir ${BASE_DIR}/output/karpathy --epochs 10 --tokenizer ai-forever/mGPT --gpt2_model ai-forever/mGPT --bs 16
