#!/bin/sh
set -e

MSG_PREFIX=[LOG_MSG]
BASE_DIR=reformulation_experiment/de
EXP_IND=0

echo "German captioning with additional datasets, experiment ${EXP_IND}"

# Base training
echo "$MSG_PREFIX Prepare base training data"
venv2/bin/python ${BASE_DIR}/prepare_base_training_data2.py ${EXP_IND}
echo "$MSG_PREFIX base preprocess"
rm -f data/coco/multi30k_train_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/base_train_data/multi30k_train_data_${EXP_IND}.json --output_file multi30k_train_data_${EXP_IND}
echo "$MSG_PREFIX Base training"
venv2/bin/python train.py --data ./data/coco/multi30k_train_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_base --epochs 10 --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Base inference"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --split test --output_file ${BASE_DIR}/data/infer/base_infer_on_test_${EXP_IND} --gpt2_model dbmdz/german-gpt2

# Translation based training
echo "$MSG_PREFIX Prepare translation training data"
venv2/bin/python ${BASE_DIR}/prepare_translation_training_data2.py ${EXP_IND}
echo "$MSG_PREFIX Translation preprocess"
rm -f data/coco/multi30k_additional_train_translated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/multi30k_additional_train_translated_data_${EXP_IND}.json --output_file multi30k_additional_train_translated_data_${EXP_IND}
echo "$MSG_PREFIX Translation training"
venv2/bin/python train.py --data ./data/coco/multi30k_additional_train_translated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_translated --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Translation inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_translated/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/translated_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Own captions based training: COCO
echo "$MSG_PREFIX Base inference on additional train: COCO"
venv2/bin/python inference.py --json_file ${BASE_DIR}/data/image_ids/coco_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/data/infer/base_infer_on_coco_${EXP_IND} --dataset COCO --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own data preperation COCO"
venv2/bin/python ${BASE_DIR}/prepare_own_training_data_coco.py ${EXP_IND}
echo "$MSG_PREFIX Own captions preprocess COCO"
rm -f data/coco/multi30k_coco_generated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_coco_${EXP_IND}.json --output_file multi30k_coco_generated_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training COCO"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own_coco --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own captions inference 1 epoch COCO"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/own_coco_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own captions inference 5 epochs COCO"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/own_coco_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

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
echo "$MSG_PREFIX Reformulations inference 1 epoch COCO"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_coco_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference 5 epochs COCO"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_coco_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Own captions based training: COCO+AIC
echo "$MSG_PREFIX Base inference on additional train: AIC"
venv2/bin/python inference.py --json_file ${BASE_DIR}/data/image_ids/all_aic_image_ids_${EXP_IND}.json --model_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --output_file ${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND} --dataset aic --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own data preperation COCO+AIC"
venv2/bin/python ${BASE_DIR}/prepare_own_training_data_coco_aic.py ${EXP_IND}
echo "$MSG_PREFIX Own captions preprocess COCO+AIC"
rm -f data/coco/multi30k_coco_aic_generated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/own_train_data/own_train_data_coco_aic_${EXP_IND}.json --output_file multi30k_coco_aic_generated_data_${EXP_IND}
echo "$MSG_PREFIX Own captions training COCO+AIC"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_aic_generated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_own_coco_aic --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own captions inference 1 epoch COCO+AIC"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco_aic/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/own_coco_aic_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Own captions inference 5 epochs COCO+AIC"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_own_coco_aic/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/own_coco_aic_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# Reformulations based training: COCO+AIC
echo "$MSG_PREFIX de->en: AIC"
venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}.json --output_file ${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}_en_translated --source_language de --target_language en --output_format caption
echo "$MSG_PREFIX Reformulation AIC"
cd ../AliceMind/mPLUG
rm -f ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}_en_reformulated.json
venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}_en_translated.json --output_format caption --output_file ../../CLIP_prefix_caption/${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}_en_reformulated --dataset aic
cd ../../CLIP_prefix_caption
echo "$MSG_PREFIX en->de AIC"
venv2/bin/python translate.py --input_file ${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}_en_reformulated.json --output_file ${BASE_DIR}/data/infer/base_infer_on_aic_${EXP_IND}_reformulated --source_language en --target_language de --output_format caption
echo "$MSG_PREFIX Reformulations data preperation COCO+AIC"
venv2/bin/python ${BASE_DIR}/prepare_reformulation_training_data_coco_aic.py ${EXP_IND}
echo "$MSG_PREFIX Reformulations preprocess COCO+AIC"
rm -f data/coco/multi30k_coco_aic_reformulated_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/re_train_data/reformulations_train_data_coco_aic_${EXP_IND}.json --output_file multi30k_coco_aic_reformulated_data_${EXP_IND}
echo "$MSG_PREFIX Reformulations training COCO+AIC"
venv2/bin/python train.py --data ./data/coco/multi30k_coco_aic_reformulated_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco_aic --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference 1 epoch COCO+AIC"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco_aic/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_coco_aic_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX Reformulations inference 5 epochs COCO+AIC"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_reformulated_coco_aic/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/reformulations_coco_aic_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# NOT IMPLEMENTED
: '
# mPLUG based training
echo "$MSG_PREFIX mPLUG data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/data/translated_data/flickr30k_mplug_de_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/mplug_train_data_${EXP_IND}.json ${EXP_IND}
echo "$MSG_PREFIX mPLUG preprocess"
rm -f data/coco/multi30k_val_mplug_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/mplug_train_data_${EXP_IND}.json --output_file multi30k_val_mplug_data_${EXP_IND}
echo "$MSG_PREFIX mPLUG training"
venv2/bin/python train.py --data ./data/coco/multi30k_val_mplug_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_mplug --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/mplug_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/mplug_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# mPLUG re based training
echo "$MSG_PREFIX mPLUG re data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/data/translated_data/flickr30k_mplug_re_de_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/mplug_re_train_data_${EXP_IND}.json ${EXP_IND}
echo "$MSG_PREFIX mPLUG re preprocess"
rm -f data/coco/multi30k_val_mplug_re_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/mplug_re_train_data_${EXP_IND}.json --output_file multi30k_val_mplug_re_data_${EXP_IND}
echo "$MSG_PREFIX mPLUG re training"
venv2/bin/python train.py --data ./data/coco/multi30k_val_mplug_re_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG re inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/mplug_re_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX mPLUG re inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_mplug_re/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/mplug_re_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2

# BLIP based training
echo "$MSG_PREFIX BLIP data preperation"
venv2/bin/python ${BASE_DIR}/convert_to_training_data.py ${BASE_DIR}/data/translated_data/flickr30k_blip_de_translated_helsinki.json ${BASE_DIR}/data/translated_train_data/blip_train_data_${EXP_IND}.json ${EXP_IND}
echo "$MSG_PREFIX BLIP preprocess"
rm -f data/coco/multi30k_val_blip_data_${EXP_IND}_tokens.pkl
venv2/bin/python parse_coco.py --clip_model_type ViT-B/32 --json_file ${BASE_DIR}/data/translated_train_data/blip_train_data_${EXP_IND}.json --output_file multi30k_val_blip_data_${EXP_IND}
echo "$MSG_PREFIX BLIP training"
venv2/bin/python train.py --data ./data/coco/multi30k_val_blip_data_${EXP_IND}.pkl --out_dir ${BASE_DIR}/output/exp_${EXP_IND}_blip --epochs 5 --load_model_from_path ${BASE_DIR}/output/exp_${EXP_IND}_base/coco_prefix-009.pt --tokenizer dbmdz/german-gpt2 --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX BLIP inference 1 epoch"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_blip/coco_prefix-000.pt --split test --output_file ${BASE_DIR}/data/infer/blip_infer_on_test_${EXP_IND}_1_epoch --gpt2_model dbmdz/german-gpt2
echo "$MSG_PREFIX BLIP inference 5 epochs"
venv2/bin/python inference.py --dataset flickr30k --model_path ${BASE_DIR}/output/exp_${EXP_IND}_blip/coco_prefix-004.pt --split test --output_file ${BASE_DIR}/data/infer/blip_infer_on_test_${EXP_IND}_5_epoch --gpt2_model dbmdz/german-gpt2'

echo "Finished!"
