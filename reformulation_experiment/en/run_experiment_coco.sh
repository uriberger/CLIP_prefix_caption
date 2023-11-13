#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/en
EXP_IND=2
SAMPLE_NUM=100000
VAL_SAMPLE_NUM=1000
VAL_STRING="--validation_set_path data/coco/val_data_${EXP_IND}.pkl --steps_evaluation 20"
BASE_SAMPLE_NUM=50000
RE_NUM=2

echo "Reformulation training on COCO with additional training on COCO, experiment ${EXP_IND}, ${SAMPLE_NUM} samples, ${BASE_SAMPLE_NUM} base samples, re num ${RE_NUM}"

: '# Prepare data
echo "$MSG_PREFIX Prepare data"
venv2/bin/python ${BASE_DIR}/prepare_coco_training_data.py ${EXP_IND} ${SAMPLE_NUM} ${VAL_SAMPLE_NUM}
echo "MSG_PREFIX val data preprocess"
rm -f data/coco/val_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/train_data/val_data_${EXP_IND}.json --output_file val_data_${EXP_IND}

# GT based training
echo "$MSG_PREFIX GT preprocess"
rm -f data/coco/gt_train_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/train_data/gt_train_data_${EXP_IND}.json --output_file gt_train_data_${EXP_IND}
echo "$MSG_PREFIX GT training"
venv2/bin/python train.py --data ./data/coco/gt_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_gt --epochs 1 ${VAL_STRING}
echo "$MSG_PREFIX GT inference on COCO"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/gt_infer_on_coco_test_${EXP_IND}
echo "$MSG_PREFIX GT inference on Flickr30k"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/gt_infer_on_flickr_test_${EXP_IND}
echo "$MSG_PREFIX GT inference on XM3600"
venv2/bin/python inference.py --dataset XM3600 --model_path ${BASE_DIR}/output/exp_${EXP_IND}_gt/coco_prefix-000.pt --output_file ${BASE_DIR}/data/infer/gt_infer_on_xm3600_${EXP_IND}

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_coco_re_training_data.py ${EXP_IND} ${BASE_SAMPLE_NUM}
echo "$MSG_PREFIX base preprocess"
rm -f data/coco/base_train_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/train_data/base_train_data_${EXP_IND}.json --output_file base_train_data_${EXP_IND}
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/base_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 1 ${VAL_STRING}'

# Re training
echo "$MSG_PREFIX Re data splitting"
venv2/bin/python ${BASE_DIR}/split_re_data.py ${EXP_IND} ${RE_NUM}
for (( i=0; i<$RE_NUM; i++ ))
do
    if [ $i -eq 0 ]
    then
       PREV_MODEL_PATH="${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-000.pt"
    else
       PREV_I=${i-1}
       PREV_MODEL_PATH="${BASE_DIR}/output/exp_${EXP_IND}_re_${PREV_I}/coco_prefix-000.pt"
    fi
    echo "$MSG_PREFIX Base inference on additional $i"
    venv2/bin/python inference.py --json_file ${BASE_DIR}/data/train_data/additional_train_image_ids_${EXP_IND}_${i}.json --model_path ${PREV_MODEL_PATH} --output_file ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_${i} --dataset COCO
    echo "$MSG_PREFIX Reformulation $i"
    rm -f ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_${i}_reformulated.json
    cd ../AliceMind/mPLUG
    venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_${i}.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_${i}_reformulated --dataset COCO
    cd ../../CLIP_prefix_caption
    echo "$MSG_PREFIX Re data preparation $i"
    venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/data/infer/base_infer_on_additional_train_${EXP_IND}_${i}_reformulated.json ${BASE_DIR}/data/train_data/re_train_data_${EXP_IND}_${i}.json COCO
    echo "$MSG_PREFIX Re preprocess $i"
    rm -f data/coco/re_train_data_${EXP_IND}_${i}_tokens.pkl
    venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/train_data/re_train_data_${EXP_IND}_${i}.json --output_file re_train_data_${EXP_IND}_${i}
    echo "$MSG_PREFIX Re training $i"
    venv2/bin/python train.py --data ./data/coco/re_train_data_${EXP_IND}_${i}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_re_${i} --epochs 1 --load_model_from_path ${PREV_MODEL_PATH} ${VAL_STRING}
    echo "$MSG_PREFIX delete previous model $i"
    if [ $i -gt 0 ]
    then
       PREV_I=${i-1}
       DEL_STR="rm -f ${BASE_DIR}/output/exp_${EXP_IND}_re_${PREV_I}/*.pt"
       echo "$MSG_PREFIX running: $DEL_STR"
       eval "$DEL_STR"
    fi
done
LAST_IND=${RE_NUM-1}
RE_MODEL_PATH="${BASE_DIR}/output/exp_${EXP_IND}_re_${LAST_IND}/coco_prefix-000.pt"
echo "$MSG_PREFIX Re inference on COCO"
venv2/bin/python inference.py --dataset COCO --model_path ${RE_MODEL_PATH} --split test --output_file ${BASE_DIR}/data/infer/re_infer_on_coco_test_${EXP_IND}
echo "$MSG_PREFIX Re inference on Flickr30k"
venv2/bin/python inference.py --dataset flickr30k --model_path ${RE_MODEL_PATH} --split test --output_file ${BASE_DIR}/data/infer/re_infer_on_flickr_test_${EXP_IND}
echo "$MSG_PREFIX Re inference on XM3600"
venv2/bin/python inference.py --dataset XM3600 --model_path ${RE_MODEL_PATH} --output_file ${BASE_DIR}/data/infer/re_infer_on_xm3600_${EXP_IND}

: '# Translation based training
echo "$MSG_PREFIX Prepare translation training data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data2.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
rm -f data/coco/coco_val_translated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/coco_val_translated_data_${EXP_IND}.json --output_file coco_val_translated_data_${EXP_IND}
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ./data/coco/coco_val_translated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX Translation inference 1 epoch"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Translation inference 5 epochs"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_5_epoch

# Own captions based training
echo "$MSG_PREFIX Base inference on val"
venv2/bin/python inference.py --json_file ${BASE_DIR}/data/image_ids/val_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --split val --output_file ${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND} --dataset COCO
echo "$MSG_PREFIX Own data preperation"
venv2/bin/python ${BASE_DIR}/prepare_own_training_data.py ${EXP_IND}
echo "$MSG_PREFIX Own captions preprocess"
rm -f data/coco/coco_val_generated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_${EXP_IND}.json --output_file coco_val_generated_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training"
venv2/bin/python train.py --data ./data/coco/coco_val_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX Own captions inference 1 epoch"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Own captions inference 5 epochs"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/own_infer_on_test_${EXP_IND}_5_epoch

# Reformulations based training
cd ../AliceMind/mPLUG
echo "$MSG_PREFIX Reformulation"
rm -f ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_reformulated.json
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_val_${EXP_IND}_reformulated --dataset COCO
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX Reformulations data preperation"
venv2/bin/python ${BASE_DIR}/prepare_reformulation_training_data2.py ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess"
rm -f data/coco/coco_val_reformulated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data/reformulations_train_data_${EXP_IND}.json --output_file coco_val_reformulated_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training"
venv2/bin/python train.py --data ./data/coco/coco_val_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt
echo "$MSG_PREFIX Reformulations inference 1 epoch"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_1_epoch
echo "$MSG_PREFIX Reformulations inference 5 epochs"
venv2/bin/python inference.py --dataset COCO --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_infer_on_test_${EXP_IND}_5_epoch'
