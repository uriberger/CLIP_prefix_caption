#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/zh
EXP_IND=0

: '# Base training
echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_base_training_data.py ${EXP_IND}
echo "$MSG_PREFIX base preprocess"
DATA_FILE=data/coco/base_train_data.pkl
if [ -f $DATA_FILE ]; then
    echo "Base data file exists."
else
    echo "Base data file doesnt exist. Creating"
    rm -f data/coco/base_train_data_tokens.pkl
    venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/base_train_data/train_data.json --output_file base_train_data
fi
echo "$MSG_PREFIX Base training"'
venv2/bin/python train.py --data ./data/coco/base_train_data.pkl --out_dir /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --tokenizer ckiplab/gpt2-base-chinese --gpt2_model ckiplab/gpt2-base-chinese
echo "$MSG_PREFIX Base inference"
venv2/bin/python inference.py --dataset aic --model_path /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --split test --output_file ${BASE_DIR}/data/infer/base_infer_on_test_${EXP_IND} --gpt2_model ckiplab/gpt2-base-chinese

# Translation based training
: 'echo "$MSG_PREFIX Prepare translation training data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
rm -f data/coco/multi30k_val_translated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/multi30k_val_translated_data_${EXP_IND}.json --output_file multi30k_val_translated_data_${EXP_IND}
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ./data/coco/multi30k_val_translated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2'

# Own captions based training
echo "$MSG_PREFIX Base inference on val"
#venv2/bin/python inference.py --json_file ${BASE_DIR}/data/image_ids/val_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND} --dataset flickr30k --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own data preperation"
#venv2/bin/python ${BASE_DIR}/prepare_own_training_data.py ${EXP_IND}
echo "$MSG_PREFIX Own captions preprocess"
#rm -f data/coco/multi30k_val_generated_data_${EXP_IND}_tokens.pkl
#venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_${EXP_IND}.json --output_file multi30k_val_generated_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training"
#venv2/bin/python train.py --data ./data/coco/multi30k_val_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own captions inference 1 epoch"
#venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own captions inference 5 epochs"
#venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Reformulations based training
echo "$MSG_PREFIX de->en"
#venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_en_translated --source_language de --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation"
cd ../AliceMind/mPLUG
#rm -f ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_en_reformulated.json
#venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_en_translated.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_en_reformulated --dataset flickr30k
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX en->de"
#venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_reformulated --source_language en --target_language de --output_format caption
echo "$MSG_PREFIX Reformulations data preperation"
#venv2/bin/python ${BASE_DIR}/prepare_reformulation_training_data.py ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess"
#rm -f data/coco/multi30k_val_reformulated_data_${EXP_IND}_tokens.pkl
#venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data/reformulations_train_data_${EXP_IND}.json --output_file multi30k_val_reformulated_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training"
#venv2/bin/python train.py --data ./data/coco/multi30k_val_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference 1 epoch"
#venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference 5 epochs"
#venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2
