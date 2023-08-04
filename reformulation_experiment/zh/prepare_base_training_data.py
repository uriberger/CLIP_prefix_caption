import json
import sys
import random
from collections import defaultdict

assert len(sys.argv) == 4
exp_ind = int(sys.argv[1])
train_sample_num = int(sys.argv[2])
additional_train_sample_num = int(sys.argv[2])

with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r') as fp:
    aic_train_data = json.load(fp)
with open('../coco-caption/annotations/aic_test_gt.json', 'r') as fp:
    aic_test_data = json.load(fp)
    test_image_ids = [x['id'] for x in aic_test_data['images']]
cur_train_data = random.sample(aic_train_data, train_sample_num)
train_image_ids = [int(x['image_id'].split('.jpg')[0], 16) for x in cur_train_data]
image_ids_dict = {x: True for x in train_image_ids}

res = [{'image_id': int(x['image_id'].split('.jpg')[0], 16), 'image_path': f'/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_images_20170902/{x["image_id"]}', 'sentences': [{'raw': y} for y in x['caption']]} for x in cur_train_data]
    
with open(f'reformulation_experiment/zh/data/base_train_data/train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(res))

not_used_train_image_ids = [int(x['image_id'].split('.jpg')[0], 16) for x in aic_train_data if int(x['image_id'].split('.jpg')[0], 16) not in image_ids_dict]
additional_train_image_ids = random.sample(not_used_train_image_ids, additional_train_sample_num)

with open(f'reformulation_experiment/zh/data/image_ids/additional_train_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(additional_train_image_ids))
