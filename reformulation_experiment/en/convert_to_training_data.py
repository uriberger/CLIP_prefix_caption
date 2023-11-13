import json
import os
import sys
from collections import defaultdict

assert len(sys.argv) == 4
input_file = sys.argv[1]
output_file = sys.argv[2]
dataset = sys.argv[3]
assert dataset in ['COCO']

with open(input_file, 'r') as fp:
    data = json.load(fp)

data_dict = defaultdict(list)
for x in data:
    data_dict[x['image_id']].append(x['caption'])

if dataset == 'COCO':
    with open('dataset_coco.json', 'r') as fp:
        coco_data = json.load(fp)['images']
    iid_to_split = {x['cocoid']: 'train' if x['split'] == 'train' else 'val' for x in coco_data}
    iid_to_path = lambda x:f'/cs/labs/oabend/uriber/datasets/COCO/{iid_to_split[x]}2014/COCO_{iid_to_split[x]}2014_{str(x).zfill(12)}.jpg'

res = [{'image_path': iid_to_path(x[0]), 'image_id': x[0], 'sentences': [{'raw': y} for y in x[1]]} for x in data_dict.items()]
with open(output_file, 'w') as fp:
    fp.write(json.dumps(res))
