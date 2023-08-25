import json
import sys
import random
from collections import defaultdict

with open(f'reformulation_experiment/de/data/infer/base_infer_on_coco_{sys.argv[1]}_reformulated.json', 'r') as fp:
    data = json.load(fp)

image_id_to_split = {}
with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
    for sample in coco_data:
        image_id = sample['cocoid']
        image_id_to_split[image_id] = sample['filepath'].split('2014')[0]
    
res = []
for sample in data:
    image_id = sample['image_id']
    split = image_id_to_split[image_id]
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'sentences': [{'raw': sample['caption']}]})
    
with open(f'reformulation_experiment/de/data/re_train_data/reformulations_train_data_coco_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
