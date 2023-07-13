#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment
EXP_IND=0

# Base training on multi30k
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/multi30k.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Base inference"
venv2/bin/python inference.py multi30k ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
mv res_multi30k.json ${BASE_DIR}/data/infer/base_infer_on_test_${EXP_IND}.json

# Translation based training
echo "$MSG_PREFIX Prepare translation training data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_data/coco_translated_data_${EXP_IND}.json --output_file coco_translated_data_${EXP_IND}
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ./data/coco/coco_translated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 1 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference"
venv2/bin/python inference.py multi30k ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt dbmdz/german-gpt2
mv res_multi30k.json ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}.json

# Reformulation based training
echo "$MSG_PREFIX Base inference on train"
venv2/bin/python inference.py ${BASE_DIR}/data/image_ids/image_ids_${EXP_IND}.json ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt dbmdz/german-gpt2
mv res_image_ids_${EXP_IND}.json ${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}.json
echo "$MSG_PREFIX de -> en"
venv2/bin/python translate.py --source_language de --target_language en --input_file ${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}_en --output_format caption
cd ../AliceMind/mPLUG
echo "$MSG_PREFIX Reformulation"
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}_en.json --split train --output_format caption
mv ann.json ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}_en_reformulated.json
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX en -> de"
venv2/bin/python translate.py --source_language en --target_language de --input_file ${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}_reformulated --output_format image
echo "$MSG_PREFIX Reformulations preprocess"
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/infer/base_infer_on_train_${EXP_IND}_reformulated.json --output_file coco_reformulated_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training"
venv2/bin/python train.py --data ./data/coco/coco_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated --epochs 1 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference"
venv2/bin/python inference.py multi30k ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-000.pt dbmdz/german-gpt2
mv res_multi30k.json ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}.json
