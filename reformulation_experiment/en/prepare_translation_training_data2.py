import json
import sys
import random
from collections import defaultdict

with open('reformulation_experiment/en/data/translated_data/stair_captions_val_en_translated_helsinki.json', 'r') as fp:
    stair_en_data = json.load(fp)

image_id_to_captions = defaultdict(list)
for sample in stair_en_data:
    image_id_to_captions[sample['image_id']].append(sample['caption'])

res = []
for image_id, captions in image_id_to_captions.items():
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg', 'sentences': [{'raw': random.choice(captions)}]})
    
with open(f'reformulation_experiment/en/data/translated_data/coco_val_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
