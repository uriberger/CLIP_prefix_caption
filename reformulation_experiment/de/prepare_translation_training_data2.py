import json
import sys
from collections import defaultdict

with open('reformulation_experiment/de/data/translated_data/coco_de_translated_google.json', 'r') as fp:
    coco_de_data = json.load(fp)

image_id_to_captions = defaultdict(list)
for sample in coco_de_data:
    image_id_to_captions[sample['image_id']].append(sample['caption'])

image_id_to_split = {}
with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
    for sample in coco_data:
        image_id = sample['cocoid']
        image_id_to_split[image_id] = sample['filepath'].split('2014')[0]

res = []
for image_id, captions in image_id_to_captions.items():
    split = image_id_to_split[image_id]
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg', 'sentences': [{'raw': x} for x in captions]})
    
with open(f'reformulation_experiment/de/data/translated_train_data/multi30k_additional_train_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
