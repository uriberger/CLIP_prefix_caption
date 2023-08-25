import json
import sys
import random
from collections import defaultdict

with open('reformulation_experiment/de/data/translated_data/coco_de_translated_helsinki.json', 'r') as fp:
    coco_de_data = json.load(fp)

image_id_to_captions = defaultdict(list)
for sample in coco_de_data:
    image_id_to_captions[sample['image_id']].append(sample['caption'])

res = []
for image_id, captions in image_id_to_captions.items():
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg', 'sentences': [{'raw': random.choice(captions)}]})
    
with open(f'reformulation_experiment/de/data/translated_train_data/multi30k_additional_train_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
