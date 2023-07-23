import json
import sys
import random

with open('dataset_coco.json', 'r') as fp:
    coco_val_data = json.load(fp)['images']
coco_val_data = [x for x in coco_val_data if x['split'] not in ['train', 'test']]

res = []
for x in coco_val_data:
    res.append({'image_id': x['cocoid'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_{str(x["cocoid"]).zfill(12)}.jpg', 'sentences': [{'raw': random.choice([y['raw'] for y in x['sentences']])}]})
    
with open(f'reformulation_experiment/en/data/gt_train_data/coco_val_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))

with open(f'reformulation_experiment/en/data/image_ids/image_ids_2.{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps([x['image_id'] for x in res]))
