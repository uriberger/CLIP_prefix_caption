import json
import os
import random
import sys

assert len(sys.argv) == 4
exp_ind = sys.argv[1]
sample_num = int(sys.argv[2])
val_sample_num = int(sys.argv[3])

with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']

coco_train_data = [x for x in coco_data if x['split'] != 'test']
sampled_data = random.sample(coco_train_data, sample_num)
sampled_data_ids = {x['cocoid']: True for x in sampled_data}
gt_data = [{'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{sampled_data[i]["filepath"]}/{sampled_data[i]["filename"]}', 'image_id': sampled_data[i]['cocoid'], 'sentences': [{'raw': random.choice(sampled_data[i]['sentences'])['raw']}]} for i in range(len(sampled_data))]
with open(f'reformulation_experiment/en/data/train_data/gt_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(gt_data))

non_sampled_data = [x for x in coco_train_data if x['cocoid'] not in sampled_data_ids]
val_sampled_data = random.sample(non_sampled_data, val_sample_num)
val_data = [{'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{x["filepath"]}/{x["filename"]}', 'image_id': x['cocoid'], 'sentences': [{'raw': random.choice(x['sentences'])['raw']}]} for x in val_sampled_data]
with open(f'reformulation_experiment/en/data/train_data/val_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(val_data))
