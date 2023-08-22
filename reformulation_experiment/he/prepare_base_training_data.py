import json
import sys
import random
from collections import defaultdict
from utils import get_cross_modal_data

assert len(sys.argv) == 3
exp_ind = int(sys.argv[1])
base_train_sample_num = int(sys.argv[2])

cross_modal_data = get_cross_modal_data()
image_ids = [x['image_id'] for x in cross_modal_data]
base_train_image_ids = random.sample(image_ids, base_train_sample_num)
base_train_image_ids_dict = {x: True for x in base_train_image_ids}
test_image_ids = [x for x in image_ids if x not in base_train_image_ids_dict]

base_train_data = [x for x in cross_modal_data if x['image_id'] in base_train_image_ids_dict]

with open(f'reformulation_experiment/he/data/base_train_data/base_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(base_train_data))

with open(f'reformulation_experiment/he/data/image_ids/test_image_ids_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(test_image_ids))
