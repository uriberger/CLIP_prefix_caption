import json
import sys
import random

with open(f'reformulation_experiment/en/data/image_ids/val_image_ids_{sys.argv[1]}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

with open('dataset_coco.json', 'r') as fp:
    coco_val_data = json.load(fp)['images']
coco_val_data = [x for x in coco_val_data if x['cocoid'] in image_ids_dict]

res = []
for x in coco_val_data:
    split = x['filepath'].split('2014')[0]
    res.append({'image_id': x['cocoid'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(x["cocoid"]).zfill(12)}.jpg', 'sentences': [{'raw': random.choice([y['raw'] for y in x['sentences']])}]})
    
with open(f'reformulation_experiment/en/data/gt_train_data/coco_val_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
