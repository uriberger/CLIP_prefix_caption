import json
import sys
import random
from collections import defaultdict

with open('reformulation_experiment/en/data/translated_data/multi30k_train_en_translated_helsinki.json', 'r') as fp:
    multi30k_en_data = json.load(fp)

image_id_to_captions = defaultdict(list)
for sample in multi30k_en_data:
    image_id_to_captions[sample['image_id']].append(sample['caption'])

res = []
for image_id, captions in image_id_to_captions.items():
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg', 'sentences': [{'raw': random.choice(captions)}]})
    
with open(f'reformulation_experiment/en/data/translated_data/multi30k_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))

with open(f'reformulation_experiment/en/data/image_ids/image_ids_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps([x['image_id'] for x in res]))
