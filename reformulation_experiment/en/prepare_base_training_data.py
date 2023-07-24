import json
import sys
import random

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
train_sample_num = int(sys.argv[2])

with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
coco_train_data = [x for x in coco_data if x['split'] == 'train']
all_image_ids = [x['cocoid'] for x in coco_train_data]
chosen_image_ids = random.sample(all_image_ids, train_sample_num)
image_ids_dict = {x: True for x in chosen_image_ids}
cur_train_data = [x for x in coco_train_data if x['cocoid'] in image_ids_dict]

res = [{'image_id': x['cocoid'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(x["cocoid"]).zfill(12)}.jpg', 'sentences': x['sentences']} for x in cur_train_data]
    
with open(f'reformulation_experiment/en/data/base_train_data/coco_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(res))

train_and_val_image_ids = [x['cocoid'] for x in coco_data if x['split'] != 'test']
val_image_ids = [x for x in train_and_val_image_ids if x not in image_ids_dict]

with open(f'reformulation_experiment/en/data/image_ids/val_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(val_image_ids))
