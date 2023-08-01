import json
import sys
import random
from collections import defaultdict

with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']

with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_train.json', 'r', encoding='utf-8') as fp:
    stair_train_data = json.load(fp)['annotations']

english_train_data = [{'image_id': x['cocoid'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(x["cocoid"]).zfill(12)}.jpg', 'sentences': [{'raw': '[EN] ' + y['raw']} for y in x['sentences']]} for x in coco_data if x['split'] == 'train']

image_id_to_caption = defaultdict(list)
for sample in stair_train_data:
    image_id_to_caption[sample['image_id']].append(sample['caption'])
japanese_train_data = [{'image_id': x[0], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(x[0]).zfill(12)}.jpg', 'sentences': [{'raw': '[JA] ' + y} for y in x[1]]} for x in image_id_to_caption.items()]

train_data = english_train_data + japanese_train_data
with open('generalization_experiment/data/karpathy/train_data.json', 'w') as fp:
    fp.write(json.dumps(train_data))
