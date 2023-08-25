import json
import sys
import random
from collections import defaultdict
from utils import get_caption_data_for_split
import pickle

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])

with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
    flickr_data = json.load(fp)['images']
with open('dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']
with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r') as fp:
    aic_train_data = json.load(fp)
with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', 'r') as fp:
    aic_val_data = json.load(fp)

new_to_orig_image_ids = []
flickr_orig_to_new_image_ids = {}
coco_orig_to_new_image_ids = {}
aic_orig_to_new_image_ids = {}

for x in flickr_data:
    if x['split'] != 'test':
        image_id = int(x['filename'].split('.jpg')[0])
        flickr_orig_to_new_image_ids[image_id] = len(new_to_orig_image_ids)
        new_to_orig_image_ids.append(('flickr30k', image_id))

for x in coco_data:
    coco_orig_to_new_image_ids[x['cocoid']] = len(new_to_orig_image_ids)
    new_to_orig_image_ids.append(('COCO', x['cocoid']))

all_aic_images_ids = []
for x in aic_train_data + aic_val_data:
    image_id = int(x['image_id'].split('.jpg')[0], 16)
    aic_orig_to_new_image_ids[image_id] = len(new_to_orig_image_ids)
    new_to_orig_image_ids.append(('aic', image_id))
    all_aic_image_ids.append(image_id)

with open(f'reformulation_experiment/de/data/image_ids/new_to_orig_image_ids_{exp_ind}.pkl', 'wb') as fp:
    pickle.dump(new_to_orig_image_ids, fp)
with open(f'reformulation_experiment/de/data/image_ids/flickr_orig_to_new_image_ids_{exp_ind}.pkl', 'wb') as fp:
    pickle.dump(flickr_orig_to_new_image_ids, fp)
with open(f'reformulation_experiment/de/data/image_ids/coco_orig_to_new_image_ids_{exp_ind}.pkl', 'wb') as fp:
    pickle.dump(coco_orig_to_new_image_ids, fp)
with open(f'reformulation_experiment/de/data/image_ids/flickr_orig_to_new_image_ids_{exp_ind}.pkl', 'wb') as fp:
    pickle.dump(aic_orig_to_new_image_ids, fp)
with open(f'reformulation_experiment/de/data/image_ids/all_aic_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(all_aic_image_ids))
with open(f'reformulation_experiment/de/data/image_ids/coco_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps([x['cocoid'] for x in coco_data]))

multi30k_data = get_caption_data_for_split('train') + get_caption_data_for_split('val') + get_caption_data_for_split('test_2016')
train_image_ids = [int(x['filename'].split('.jpg')[0]) for x in flickr_data if x['split'] != 'test']
image_ids_dict = {x: True for x in chosen_image_ids}
train_data = [x for x in multi30k_data if x['image_id'] in image_ids_dict]

image_id_to_captions = defaultdict(list)
for x in cur_train_data:
    image_id_to_captions[x['image_id']].append(x['caption'])

res = [{'image_id': x[0], 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x[0]}.jpg', 'sentences': [{'raw': y} for y in x[1]]} for x in image_id_to_captions.items()]
    
with open(f'reformulation_experiment/de/data/base_train_data/multi30k_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(res))
