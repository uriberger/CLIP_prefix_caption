import sys
import os
import json
import random

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
dir_path = sys.argv[2]

with open(os.path.join(dir_path, 'translated_data', 'coco_es_translated_helsinki.json'), 'r') as fp:
    data = json.load(fp)

for i in range(len(data)):
    data[i]['sentences'] = random.sample(data[i]['sentences'], 1)

with open(os.path.join(dir_path, 'data', f'translation_train_data_{exp_ind}.json'), 'w') as fp:
    fp.write(json.dumps(data))
