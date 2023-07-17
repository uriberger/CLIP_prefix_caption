import json
import sys
import random
from collections import defaultdict

sample_num = None
with open('dataset_coco.json', 'r') as fp:
    coco_en_data = json.load(fp)['images']
data_inds = [i for i in range(len(coco_en_data)) if coco_en_data[i]['split'] == 'train']
if sample_num is not None:
    data_inds = random.sample(data_inds, sample_num)

with open('reformulation_experiment/de/data/translated_data/coco_de_translated_helsinki.json', 'r') as fp:
    coco_de_data = json.load(fp)

caption_inds = [i for i in range(len(coco_en_data)) for x in coco_en_data[i]['sentences']]
ind_to_captions = defaultdict(list)
for i in range(len(coco_de_data)):
    cur_ind = caption_inds[i]
    caption = coco_de_data[i]
    ind_to_captions[cur_ind].append(caption)

res = []
for ind in data_inds:
    res.append({'image_id': coco_en_data[ind]['cocoid'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(coco_en_data[ind]["cocoid"]).zfill(12)}.jpg', 'sentences': [{'raw': random.choice(ind_to_captions[ind])}]})
    
with open(f'reformulation_experiment/data/translated_data/coco_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))

with open(f'reformulation_experiment/data/image_ids/image_ids_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps([x['image_id'] for x in res]))
