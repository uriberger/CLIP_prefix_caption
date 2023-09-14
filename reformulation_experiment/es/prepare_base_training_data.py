import sys
import random
import os
import json

assert len(sys.argv) == 4
exp_ind = int(sys.argv[1])
base_sample_num = 3000
dir_path = sys.argv[3]

with open('crossmodal3600_es.json', 'r') as fp:
    data = json.load(fp)
    
base_train_data = random.sample(data, base_sample_num)
with open(os.path.join(dir_path, 'data', f'base_train_data_{exp_ind}.json'), 'w') as fp:
    fp.write(json.dumps(base_train_data))
    
base_train_image_ids = [x['image_id'] for x in base_train_data]
base_train_dict = {x: True for x in base_train_image_ids}
test_data = [x for x in data if x['image_id'] not in base_train_dict]
with open(os.path.join(dir_path, 'data', f'test_data_{exp_ind}.json'), 'w') as fp:
    fp.write(json.dumps(test_data))
