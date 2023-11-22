import json
import os
import random
import sys

assert len(sys.argv) == 3
exp_ind = sys.argv[1]
base_sample_num = int(sys.argv[2])

with open(f'reformulation_experiment/en/data/train_data/gt_train_data_{exp_ind}.json', 'r') as fp:
    gt_data = json.load(fp)
base_data = random.sample(gt_data, base_sample_num)
with open(f'reformulation_experiment/en/data/train_data/base_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(base_data))
base_image_ids = {x['image_id']: True for x in base_data}
additional_image_ids = [x['image_id'] for x in gt_data if x['image_id'] not in base_image_ids]
with open(f'reformulation_experiment/en/data/train_data/additional_train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(additional_image_ids))
