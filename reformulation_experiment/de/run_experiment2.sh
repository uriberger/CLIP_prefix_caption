#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/de
EXP_IND=5
BASE_SAMPLE_NUM=

echo "German captioning with additional datasets, experiment ${EXP_IND}"

# Base training
: 'echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_base_training_data2.py ${EXP_IND} ${BASE_SAMPLE_NUM}
echo "$MSG_PREFIX base preprocess"
rm -f data/coco/multi30k_train_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/base_train_data/multi30k_train_data_${EXP_IND}.json --output_file multi30k_train_data_${EXP_IND}
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/multi30k_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Base inference on cross modal"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/base_infer_on_crossmodal_${EXP_IND} --gpt2_model dbmdz/german-gpt2

# Translation based training
echo "$MSG_PREFIX Prepare translation training data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data2.py ${EXP_IND} ${BASE_DIR}
echo "$MSG_PREFIX Translation preprocess"
rm -f data/coco/multi30k_additional_train_translated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/multi30k_additional_train_translated_data_${EXP_IND}.json --output_file multi30k_additional_train_translated_data_${EXP_IND}'
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ./data/coco/multi30k_additional_train_translated_data_${EXP_IND}.pkl --out_dir /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_google_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference on cross modal 1 epoch"
venv2/bin/python inference.py --model_path /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_google_translated/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/google_translated_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference on cross modal 5 epochs"
venv2/bin/python inference.py --model_path /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_google_translated/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/google_translated_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference on flickr30k 1 epoch"
venv2/bin/python inference.py --model_path /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_google_translated/coco_prefix-000.pt --dataset flickr30k --split test --output_file ${BASE_DIR}/data/infer/google_translated_infer_on_flickr30k_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference on flickr30k 5 epochs"
venv2/bin/python inference.py --model_path /cs/labs/oabend/uriber/${BASE_DIR}/output/exp_${EXP_IND}_google_translated/coco_prefix-004.pt --dataset flickr30k --split test --output_file ${BASE_DIR}/data/infer/google_translated_infer_on_flickr30k_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Own captions based training: COCO
: 'echo "$MSG_PREFIX Base inference on additional train: COCO"
venv2/bin/python inference.py --json_file ${BASE_DIR}/data/image_ids/coco_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND} --dataset COCO --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own data preperation COCO"
venv2/bin/python ${BASE_DIR}/prepare_own_training_data_coco.py ${EXP_IND}
echo "$MSG_PREFIX Own captions preprocess COCO"
rm -f data/coco/multi30k_coco_generated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_coco_${EXP_IND}.json --output_file multi30k_coco_generated_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training COCO"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own_coco --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own inference on cross modal 1 epoch COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/own_coco_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own inference on cross modal 5 epochs COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/own_coco_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Reformulations based training: COCO
echo "$MSG_PREFIX de->en: COCO"
venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}_en_translated --source_language de --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation COCO"
cd ../AliceMind/mPLUG
rm -f ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}_en_reformulated.json
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}_en_translated.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}_en_reformulated --dataset COCO
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX en->de"
venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND}_reformulated --source_language en --target_language de --output_format caption
echo "$MSG_PREFIX Reformulations data preperation COCO"
venv2/bin/python ${BASE_DIR}/prepare_reformulation_training_data_coco.py ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess COCO"
rm -f data/coco/multi30k_coco_reformulated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data/reformulations_train_data_coco_${EXP_IND}.json --output_file multi30k_coco_reformulated_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training COCO"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference on cross modal 1 epoch COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/reformulations_coco_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference on cross modal 5 epochs COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/reformulations_coco_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# mPLUG re based training
echo "$MSG_PREFIX mPLUG re preprocess COCO"
rm -f data/coco/multi30k_additional_training_mplug_re_coco_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_data/coco_mplug_re_de_translated_helsinki.json --output_file multi30k_additional_train_mplug_re_coco_data_${EXP_IND}
echo "$MSG_PREFIX mPLUG re training COCO"
venv2/bin/python train.py --data ./data/coco/multi30k_additional_train_mplug_re_coco_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re_coco --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG re inference on cross modal 1 epoch COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re_coco/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/mplug_re_coco_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG re inference on cross modal 5 epochs COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re_coco/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/mplug_re_coco_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2'

# BLIP based training
: 'echo "$MSG_PREFIX BLIP preprocess COCO"
rm -f data/coco/multi30k_additional_training_blip_coco_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_data/coco_blip_de_translated_helsinki.json --output_file multi30k_additional_train_blip_coco_data_${EXP_IND}
echo "$MSG_PREFIX BLIP training COCO"
venv2/bin/python train.py --data ./data/coco/multi30k_additional_train_blip_coco_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_blip_coco --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX BLIP inference on cross modal 1 epoch COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_blip_coco/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/blip_coco_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX BLIP inference on cross modal 5 epochs COCO"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_blip_coco/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/blip_coco_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2'

# Own captions based training: COCO+IN
: 'echo "$MSG_PREFIX Base inference on additional train: IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND} --image_path_file ${BASE_DIR}/data/image_ids/image_net_image_paths.json --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own data preperation COCO+IN"
venv2/bin/python ${BASE_DIR}/convert_training_data_coco_in.py ${EXP_IND} ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}.json ${BASE_DIR}/data/own_train_data/own_train_data_coco_in_${EXP_IND}.json
echo "$MSG_PREFIX Own captions preprocess COCO+IN"
rm -f data/coco/multi30k_coco_in_generated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_coco_in_${EXP_IND}.json --output_file multi30k_coco_in_generated_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training COCO+IN"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_in_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own_coco_in --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own inference on cross modal 1 epoch COCO+IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco_in/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/own_coco_in_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own inference on cross modal 5 epochs COCO+IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco_in/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/own_coco_in_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Reformulations based training: COCO+IN
echo "$MSG_PREFIX de->en: IN"
venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_en_translated --source_language de --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation IN"
cd ../AliceMind/mPLUG
rm -f ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_en_reformulated.json
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_en_translated.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_en_reformulated --dataset ImageNet
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX en->de IN"
venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_reformulated --source_language en --target_language de --output_format caption
echo "$MSG_PREFIX Reformulations data preperation COCO+IN"
venv2/bin/python ${BASE_DIR}/convert_training_data_coco_in.py ${EXP_IND} ${BASE_DIR}/data/infer/base_infer_on_in_${EXP_IND}_reformulated.json ${BASE_DIR}/data/re_train_data/re_train_data_coco_in_${EXP_IND}.json
echo "$MSG_PREFIX Reformulations preprocess COCO+IN"
rm -f data/coco/multi30k_coco_in_reformulated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data/re_train_data_coco_in_${EXP_IND}.json --output_file multi30k_coco_in_reformulated_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training COCO+IN"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_in_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco_in --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference on cross modal 1 epoch COCO+IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco_in/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/reformulations_coco_in_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference on cross modal 5 epochs COCO+IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco_in/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/reformulations_coco_in_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# mPLUG re based training COCO+IN
echo "$MSG_PREFIX mPLUG re data preperation COCO+IN"
venv2/bin/python ${BASE_DIR}/convert_training_data_coco_in.py ${EXP_IND} ${BASE_DIR}/data/translated_data/in_mplug_re_de_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/mplug_re_train_data_coco_in_${EXP_IND}.json
echo "$MSG_PREFIX mPLUG re preprocess COCO+IN"
rm -f data/coco/multi30k_additional_train_mplug_re_coco_in_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/mplug_re_train_data_coco_in_${EXP_IND}.json --output_file multi30k_additional_train_mplug_re_coco_in_data
echo "$MSG_PREFIX mPLUG re training COCO+IN"
venv2/bin/python train.py --data ./data/coco/multi30k_additional_train_mplug_re_coco_in_data.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re_coco_in --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG re inference on cross modal 1 epoch COCO+IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re_coco_in/coco_prefix-000.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/mplug_re_coco_in_infer_on_crossmodal_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG re inference on cross modal 5 epochs COCO+IN"
venv2/bin/python inference.py --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re_coco_in/coco_prefix-004.pt --image_path_file crossmodal_ids_and_paths.json --output_file ${BASE_DIR}/data/infer/mplug_re_coco_in_infer_on_crossmodal_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2'

echo "Finished!"
