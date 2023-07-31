import json
import sys
import random
from collections import defaultdict

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
train_sample_num = int(sys.argv[2])

with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_train.json', 'r') as fp:
    train_data = json.load(fp)['annotations']
with open('/cs/labs/oabend/uriber/datasets/STAIR-captions/stair_captions_v1.2_val.json', 'r') as fp:
    val_data = json.load(fp)['annotations']
with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
train_image_ids = [x['cocoid'] for x in coco_data if x['split'] == 'train']
chosen_image_ids = random.sample(train_image_ids, train_sample_num)
image_ids_dict = {x: True for x in chosen_image_ids}
cur_train_data = [x for x in train_data if x['image_id'] in image_ids_dict]

image_id_to_captions = defaultdict(list)
for x in cur_train_data:
    image_id_to_captions[x['image_id']].append(x['caption'])

res = [{'image_id': x[0], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(x[0]).zfill(12)}.jpg', 'sentences': [{'raw': y} for y in x[1]]} for x in image_id_to_captions.items()]
    
with open(f'reformulation_experiment/ja/data/base_train_data/stair_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(res))

train_and_val_image_ids = [x['cocoid'] for x in coco_data if x['split'] != 'test']
val_image_ids = [x for x in train_and_val_image_ids if x not in image_ids_dict]

with open(f'reformulation_experiment/ja/data/image_ids/val_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(val_image_ids))
