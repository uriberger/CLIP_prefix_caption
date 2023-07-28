#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/en
EXP_IND=0
SAMPLE_NUM=20000

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_base_training_data_flickr.py ${EXP_IND} ${SAMPLE_NUM}
echo "$MSG_PREFIX base preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/base_train_data/flickr_train_data_${EXP_IND}.json --output_file flickr_train_data_${EXP_IND}
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/flickr_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10
rm -f data/coco/flickr_train_data_${EXP_IND}_tokens.pkl
echo "$MSG_PREFIX Base inference"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --split test --output_file ${BASE_DIR}/data/infer/base_infer_on_test_${EXP_IND}

# GT based training
echo "$MSG_PREFIX Prepare GT training data"
venv2/bin/python ${BASE_DIR}/prepare_gt_training_data_flickr.py ${EXP_IND}
echo "$MSG_PREFIX GT preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/gt_train_data/flickr_val_data_${EXP_IND}.json --output_file flickr_val_data_${EXP_IND}
rm -f data/coco/flickr_val_data_${EXP_IND}_tokens.pkl
echo "$MSG_PREFIX GT training"
venv2/bin/python train.py --data ./data/coco/flickr_val_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_gt --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX GT inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/gt_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX GT inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/gt_infer_on_test_${EXP_IND}_5_epoch

# Translation based training
echo "$MSG_PREFIX Prepare translation training data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data_flickr.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/flickr_val_translated_data_${EXP_IND}.json --output_file flickr_val_translated_data_${EXP_IND}
rm -f data/coco/flickr_val_translated_data_${EXP_IND}_tokens.pkl
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ./data/coco/flickr_val_translated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX Translation inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Translation inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_5_epoch

# Own captions based training
echo "$MSG_PREFIX Base inference on val"
venv2/bin/python inference.py --json_file ${BASE_DIR}/data/image_ids/val_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND} --dataset flickr30k
echo "$MSG_PREFIX Own data preperation"
venv2/bin/python ${BASE_DIR}/prepare_own_training_data_flickr.py ${EXP_IND}
echo "$MSG_PREFIX Own captions preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_${EXP_IND}.json --output_file flickr_val_generated_data_${EXP_IND}
rm -f data/coco/flickr_val_generated_data_${EXP_IND}_tokens.pkl
echo "$MSG_PREFIX Own captions training"
venv2/bin/python train.py --data ./data/coco/flickr_val_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX Own captions inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Own captions inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_5_epoch

# Reformulations based training
cd ../AliceMind/mPLUG
echo "$MSG_PREFIX Reformulation"
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_reformulated --dataset flickr30k
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX Reformulations data preperation"
venv2/bin/python ${BASE_DIR}/prepare_reformulation_training_data_flickr.py ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data/reformulations_train_data_${EXP_IND}.json --output_file coco_val_reformulated_data_${EXP_IND}
rm -f data/coco/coco_val_reformulated_data_${EXP_IND}_tokens.pkl
echo "$MSG_PREFIX Reformulations training"
venv2/bin/python train.py --data ./data/coco/coco_val_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX Reformulations inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Reformulations inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_5_epoch
