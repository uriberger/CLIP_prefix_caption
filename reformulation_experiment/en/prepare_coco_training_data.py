import json
import os
import random
import sys

assert len(sys.argv) == 3
exp_ind = sys.argv[1]
sample_num = sys.argv[2]

with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']

coco_train_data = [x for x in coco_data if x['split'] != 'test']
sampled_data = random.sample(coco_train_data, sample_num)
gt_data = [{'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{sampled_data[i]["file_path"]}/{sampled_data[i]["file_name"}', 'image_id': sampled_data[i]['cocoid'], 'caption': random.choice(sampled_data[i]['sentences'])['raw'], 'id': i} for i in range(len(sampled_data))]
with open(f'reformulation_experiment/en/data/train_data/gt_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(gt_data))


