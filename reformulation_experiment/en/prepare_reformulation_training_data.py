import json
import sys
import random
from collections import defaultdict

with open(f'reformulation_experiment/en/data/infer/base_infer_on_train_{sys.argv[1]}_reformulated.json', 'r') as fp:
    data = json.load(fp)

res = []
for sample in data:
    image_id = sample['image_id']
    res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg', 'sentences': [{'raw': sample['caption']}]})
    
with open(f'reformulation_experiment/en/data/re_train_data/reformulations_train_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
