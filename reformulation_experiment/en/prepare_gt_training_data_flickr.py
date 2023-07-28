import json
import sys
import random

with open(f'reformulation_experiment/en/data/image_ids/val_image_ids_{sys.argv[1]}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
    flickr_val_data = json.load(fp)['images']
flickr_val_data = [x for x in flickr_val_data if int(x['filename'].split('.jpg')[0]) in image_ids_dict]

res = []
for x in flickr_val_data:
    res.append({'image_id': int(x['filename'].split('.jpg')[0]), 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x["filename"]}', 'sentences': [{'raw': random.choice([y['raw'] for y in x['sentences']])}]})
    
with open(f'reformulation_experiment/en/data/gt_train_data/flickr_val_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
