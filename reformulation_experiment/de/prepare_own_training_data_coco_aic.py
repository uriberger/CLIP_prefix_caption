import json
import sys
import random
from collections import defaultdict

with open(f'reformulation_experiment/de/data/infer/base_infer_on_coco_{sys.argv[1]}.json', 'r') as fp:
    coco_data = json.load(fp)
with open(f'reformulation_experiment/de/data/infer/base_infer_on_aic_{sys.argv[1]}.json', 'r') as fp:
    aic_data = json.load(fp)

coco_image_id_to_split = {}
with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
    for sample in coco_data:
        image_id = sample['cocoid']
        image_id_to_split[image_id] = sample['filepath'].split('2014')[0]

aic_image_id_to_split = {}
with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r') as fp:
    aic_train_data = json.load(fp)
with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', 'r') as fp:
    aic_val_data = json.load(fp)
for sample in aic_train_data:
    image_id = int(x['image_id'].split('.jpg')[0], 16)
    image_id_to_split[image_id] = 'train'
for sample in aic_val_data:
    image_id = int(x['image_id'].split('.jpg')[0], 16)
    image_id_to_split[image_id] = 'validation'
split_to_date = {'train': '20170902', 'validation': '20170910'}
    
res = []
for sample in coco_data:
    image_id = sample['image_id']
    split = coco_image_id_to_split[image_id]
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'sentences': [{'raw': sample['caption']}]})
for sample in aic_data:
    image_id = sample['image_id']
    split = aic_image_id_to_split[image_id]
    date = split_to_date[split]
    file_name = hex(image_id)[2:].zfill(40)
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_{split}_{date}/caption_{split}_images_{date}/{file_name}', 'sentences': [{'raw': sample['caption']}]})
    
with open(f'reformulation_experiment/de/data/own_train_data/own_train_data_coco_aic_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
