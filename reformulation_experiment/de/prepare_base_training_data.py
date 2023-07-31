import json
import sys
import random
from collections import defaultdict
from utils import get_caption_data_for_split

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
train_sample_num = int(sys.argv[2])

multi30k_data = get_caption_data_for_split('train') + get_caption_data_for_split('val') + get_caption_data_for_split('test_2016')
with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
    flickr_data = json.load(fp)['images']
train_image_ids = [int(x['filename'].split('.jpg')[0]) for x in flickr_data if x['split'] == 'train']
chosen_image_ids = random.sample(train_image_ids, train_sample_num)
image_ids_dict = {x: True for x in chosen_image_ids}
cur_train_data = [x for x in multi30k_data if x['image_id'] in image_ids_dict]

image_id_to_captions = defaultdict(list)
for x in cur_train_data:
    image_id_to_captions[x['image_id']].append(x['caption'])

res = [{'image_id': x[0], 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x[0]}.jpg', 'sentences': [{'raw': y} for y in x[1]]} for x in image_id_to_captions.items()]
    
with open(f'reformulation_experiment/de/data/base_train_data/multi30k_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(res))

train_and_val_image_ids = [int(x['filename'].split('.jpg')[0]) for x in flickr_data if x['split'] != 'test']
val_image_ids = [x for x in train_and_val_image_ids if x not in image_ids_dict]

with open(f'reformulation_experiment/de/data/image_ids/val_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(val_image_ids))
