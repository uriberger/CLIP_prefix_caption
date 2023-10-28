import json
import sys
import random
from collections import defaultdict

assert len(sys.argv) == 4
in_input_file = sys.argv[2]
output_file = sys.argv[3]

with open(f'reformulation_experiment/de/data/infer/base_infer_on_coco_{sys.argv[1]}.json', 'r') as fp:
    coco_data = json.load(fp)
with open(in_input_file, 'r') as fp:
    in_data = json.load(fp)

coco_image_id_to_split = {}
with open('dataset_coco.json', 'r') as fp:
    coco_orig_data = json.load(fp)['images']
    for sample in coco_orig_data:
        image_id = sample['cocoid']
        coco_image_id_to_split[image_id] = sample['filepath'].split('2014')[0]
    
res = []
for sample in coco_data:
    image_id = sample['image_id']
    split = coco_image_id_to_split[image_id]
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'sentences': [{'raw': sample['caption']}]})
for sample in in_data:
    image_id = sample['image_id']
    res.append({'image_id': image_id, 'image_path': f'/cs/snapless/oabend/uriber/image_net/{image_id}.jpg', 'sentences': [{'raw': sample['caption']}]})
    
with open(output_file, 'w') as fp:
    fp.write(json.dumps(res))
