import json
import sys
import random
from collections import defaultdict
from utils import get_caption_data_for_split

with open(f'reformulation_experiment/de/data/image_ids/val_image_ids_{sys.argv[1]}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

multi30k_data = get_caption_data_for_split('train') + get_caption_data_for_split('val') + get_caption_data_for_split('test_2016')

image_id_to_captions = defaultdict(list)
gt_data = [x for x in multi30k_data if x['image_id'] in image_ids_dict]
for x in gt_data:
    image_id_to_captions[x['image_id']].append(x['caption'])

res = []
for x in image_id_to_captions.items():
    res.append({'image_id': x[0], 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x[0]}.jpg', 'sentences': [{'raw': random.choice(x[1])}]})
    
with open(f'reformulation_experiment/de/data/gt_train_data/multi30k_val_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
