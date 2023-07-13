import json
import random
from collections import defaultdict

sample_num = 100
with open('dataset_coco.json', 'r') as fp:
    coco_en_data = json.load(fp)['images']
data_inds = [i for i in range(len(data)) if data[i]['split'] == 'train']
if sample_num is not None:
    data_inds = random.sample(data_inds, sample_num)

with open('coco_de_translated_helskinki.json', 'r') as fp:
    coco_de_data = json.load(fp)

caption_inds = [i for i in range(len(data)) for x in data[i]['sentences']]
ind_to_captions = defaultdict(list)
for i in range(len(caption_inds)):
    cur_ind = caption_inds[i]
    caption = coco_de_data[i]
    ind_to_captions[ind].append(caption)

res = []
for ind in data_inds:
    res.append({'image_id': data[ind]['image_id'], 'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/train2014/COCO_train2014_{str(data[ind]["image_id"]).zfill(12)}.jpg', 'sentences': [{'raw': random.choice(ind_to_captions[ind])}]})
    
with open(f'reformulation_experiment/data/translated_data/coco_translated_data_{sys.argv[1]}.json', 'w') as fp:
    fp.write(json.dumps(res))
