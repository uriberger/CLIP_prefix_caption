#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/es
EXP_IND=0
MODEL_NAME=DeepESP/gpt2-spanish

echo "Spanish captioning, experiment ${EXP_IND}"

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_base_training_data.py ${EXP_IND} ${BASE_DIR}
echo "$MSG_PREFIX base preprocess"
rm -f data/coco/es_base_train_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/base_train_data_${EXP_IND}.json --output_file es_base_train_data_${EXP_IND}
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/es_base_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Base inference"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/infer/base_infer_on_test_${EXP_IND} --gpt2_model ${MODEL_NAME}

# Translation based training
echo "$MSG_PREFIX Prepare translation data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data.py ${EXP_IND} ${BASE_DIR}
echo "$MSG_PREFIX Translation preprocess"
rm -f data/coco/coco_es_translated_data_${EXP_IND}_tokens.pkl
TRANSLATED_PICKLE_FILE=data/coco/coco_es_translated_data_${EXP_IND}.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translation_train_data_${EXP_IND}.json --output_file coco_es_translated_data_${EXP_IND}
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ${TRANSLATED_PICKLE_FILE} --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Translation inference 1 epoch"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --output_file ${BASE_DIR}/infer/translated_infer_on_test_${EXP_IND}_1_epoch --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Translation inference 5 epochs"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --output_file ${BASE_DIR}/infer/translated_infer_on_test_${EXP_IND}_5_epoch --gpt2_model ${MODEL_NAME}

# Own captions based training
echo "$MSG_PREFIX Base inference on additional train"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/image_ids/coco_image_ids.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND} --dataset COCO --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Own data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py #!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/es
EXP_IND=0
MODEL_NAME=DeepESP/gpt2-spanish

echo "Spanish captioning, experiment ${EXP_IND}"

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_base_training_data.py ${EXP_IND} ${BASE_DIR}
echo "$MSG_PREFIX base preprocess"
rm -f data/coco/es_base_train_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/base_train_data_${EXP_IND}.json --output_file es_base_train_data_${EXP_IND}
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/es_base_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Base inference"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/infer/base_infer_on_test_${EXP_IND} --gpt2_model ${MODEL_NAME}

# Translation based training
echo "$MSG_PREFIX Prepare translation data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data.py ${EXP_IND} ${BASE_DIR}
echo "$MSG_PREFIX Translation preprocess"
rm -f data/coco/coco_es_translated_data_${EXP_IND}_tokens.pkl
TRANSLATED_PICKLE_FILE=data/coco/coco_es_translated_data_${EXP_IND}.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translation_train_data_${EXP_IND}.json --output_file coco_es_translated_data_${EXP_IND}
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ${TRANSLATED_PICKLE_FILE} --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Translation inference 1 epoch"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --output_file ${BASE_DIR}/infer/translated_infer_on_test_${EXP_IND}_1_epoch --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Translation inference 5 epochs"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --output_file ${BASE_DIR}/infer/translated_infer_on_test_${EXP_IND}_5_epoch --gpt2_model ${MODEL_NAME}

# Own captions based training
echo "$MSG_PREFIX Base inference on additional train"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/image_ids/coco_image_ids.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND} --dataset COCO --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Own data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND} ${BASE_DIR}/data/own_train_data_${EXP_IND}.json
echo "$MSG_PREFIX Own captions preprocess"
rm -f data/coco/coco_es_own_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data_${EXP_IND}.json --output_file coco_es_own_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training"
venv2/bin/python train.py --data ./data/coco/coco_es_own_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Own captions inference 1 epoch"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-000.pt --output_file ${BASE_DIR}/infer/own_infer_on_test_${EXP_IND}_1_epoch --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Own captions inference 5 epochs"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own/coco_prefix-004.pt --output_file ${BASE_DIR}/infer/own_infer_on_test_${EXP_IND}_5_epoch --gpt2_model ${MODEL_NAME}

# Reformulations based training
echo "$MSG_PREFIX es->en"
venv2/bin/python translate.py --input_file ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}.json --output_file ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_en_translated --source_language es --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation"
cd ../AliceMind/mPLUG
rm -f ../../CLIP_prefix_caption/${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_en_translated.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated --dataset COCO
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX en->es"
venv2/bin/python translate.py --input_file ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_reformulated --source_language en --target_language es --output_format caption
echo "$MSG_PREFIX Reformulations data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/infer/base_infer_on_additional_train_${EXP_IND}_reformulated.json ${BASE_DIR}/data/re_train_data_${EXP_IND}.json
echo "$MSG_PREFIX Reformulations preprocess"
rm -f data/coco/coco_es_re_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data_${EXP_IND}.json --output_file coco_es_re_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training"
venv2/bin/python train.py --data ./data/coco/coco_es_re_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_re --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Reformulations inference 1 epoch"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_re/coco_prefix-000.pt --output_file ${BASE_DIR}/infer/re_infer_on_test_${EXP_IND}_1_epoch --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX Reformulations inference 5 epochs"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_re/coco_prefix-004.pt --output_file ${BASE_DIR}/infer/re_infer_on_test_${EXP_IND}_5_epoch --gpt2_model ${MODEL_NAME}

# mPLUG re based training
echo "$MSG_PREFIX mPLUG re data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/translated_data/coco_mplug_re_es_translated_helsinki.json ${BASE_DIR}/data/mplug_re_train_data_${EXP_IND}.json ${EXP_IND}
echo "$MSG_PREFIX mPLUG re preprocess"
rm -f data/coco/coco_es_mplug_re_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/mplug_re_train_data_${EXP_IND}.json --output_file coco_es_mplug_re_data_${EXP_IND}
echo "$MSG_PREFIX mPLUG re training"
venv2/bin/python train.py --data ./data/coco/coco_es_mplug_re_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer ${MODEL_NAME} --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX mPLUG re inference 1 epoch"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re/coco_prefix-000.pt --output_file ${BASE_DIR}/infer/mplug_re_infer_on_test_${EXP_IND}_1_epoch --gpt2_model ${MODEL_NAME}
echo "$MSG_PREFIX mPLUG re inference 5 epochs"
venv2/bin/python inference.py --image_path_file ${BASE_DIR}/data/test_data_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re/coco_prefix-004.pt --output_file ${BASE_DIR}/infer/mplug_re_infer_on_test_${EXP_IND}_5_epoch --gpt2_model ${MODEL_NAME}
