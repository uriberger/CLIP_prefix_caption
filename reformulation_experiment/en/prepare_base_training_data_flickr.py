import json
import sys
import random

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
train_sample_num = int(sys.argv[2])

with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
    flickr_data = json.load(fp)['images']
flickr_train_data = [x for x in flickr_data if x['split'] == 'train']
all_image_ids = [int(x['filename'].split('.jpg')[0]) for x in flickr_train_data]
chosen_image_ids = random.sample(all_image_ids, train_sample_num)
image_ids_dict = {x: True for x in chosen_image_ids}
cur_train_data = [x for x in flickr_train_data if int(x['filename'].split('.jpg')[0]) in image_ids_dict]

res = [{'image_id': int(x['filename'].split('.jpg')[0]), 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x["filename"]}', 'sentences': x['sentences']} for x in cur_train_data]
    
with open(f'reformulation_experiment/en/data/base_train_data/flickr_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(res))

train_and_val_image_ids = [int(x['filename'].split('.jpg')[0]) for x in flickr_data if x['split'] != 'test']
val_image_ids = [x for x in train_and_val_image_ids if x not in image_ids_dict]

with open(f'reformulation_experiment/en/data/image_ids/val_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(val_image_ids))
