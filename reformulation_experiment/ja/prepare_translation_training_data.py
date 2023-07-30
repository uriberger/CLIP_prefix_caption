import json
import sys
import random
from collections import defaultdict

with open('reformulation_experiment/ja/data/translated_data/coco_ja_translated_helsinki.json', 'r') as fp:
    coco_ja_data = json.load(fp)

with open(f'reformulation_experiment/ja/data/image_ids/val_image_ids_{sys.argv[1]}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

image_id_to_captions = defaultdict(list)
for sample in coco_ja_data:
    if sample['image_id'] in image_ids_dict:
        image_id_to_captions[sample['image_id']].append(sample['caption'])

image_id_to_split = {}
with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
for sample in coco_data:
    image_id = sample['cocoid']
    if image_id in image_ids_dict:
        image_id_to_split[image_id] = sample['filepath'].split('2014')[0]

res = []
for image_id, captions in image_id_to_captions.items():
    split = image_id_to_split[image_id]
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'sentences': [{'raw': random.choice(captions)}]})
    
with open(f'reformulation_experiment/ja/data/translated_train_data/stair_val_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
