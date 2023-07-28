import json
import sys
import random
from collections import defaultdict

with open('reformulation_experiment/en/data/translated_data/multi30k_en_translated_helsinki.json', 'r') as fp:
    stair_en_data = json.load(fp)

with open(f'reformulation_experiment/en/data/image_ids/val_image_ids_{sys.argv[1]}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

image_id_to_captions = defaultdict(list)
for sample in stair_en_data:
    if sample['image_id'] in image_ids_dict:
        image_id_to_captions[sample['image_id']].append(sample['caption'])

res = []
for image_id, captions in image_id_to_captions.items():
    split = image_id_to_split[image_id]
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg', 'sentences': [{'raw': random.choice(captions)}]})
    
with open(f'reformulation_experiment/en/data/translated_train_data/flickr_val_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
