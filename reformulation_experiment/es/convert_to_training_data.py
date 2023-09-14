import sys
import os
from collections import defaultdict
import json

assert len(sys.argv) == 3
input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as fp:
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
    
with open(output_file, 'w') as fp:
    fp.write(json.dumps(res))
