import json
import sys
import random
from collections import defaultdict

with open(f'reformulation_experiment/ja/data/image_ids/val_image_ids_{sys.argv[1]}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_train.json', 'r') as fp:
    train_data = json.load(fp)['annotations']
with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_val.json', 'r') as fp:
    val_data = json.load(fp)['annotations']

image_id_to_captions = defaultdict(list)
    
gt_data_train = [x for x in train_data if x['image_id'] in image_ids_dict]
for x in gt_data_train:
    image_id_to_captions[x['image_id']].append(x['caption'])

gt_data_val = [x for x in val_data if x['image_id'] in image_ids_dict]
for x in gt_data_train:
    image_id_to_captions[x['image_id']].append(x['caption'])

with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
image_id_to_split = {x['cocoid']: x['filepath'].split('2014')[0] for x in coco_data}

res = []
for x in image_id_to_captions.items():
    split = image_id_to_split[x[0]]
    res.append({'image_id': x[0], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(x[0]).zfill(12)}.jpg', 'sentences': [{'raw': random.choice(x[1])}]})
    
with open(f'reformulation_experiment/ja/data/gt_train_data/stair_val_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
