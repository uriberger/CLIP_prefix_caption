import json
import sys
import random

assert len(sys.argv) == 4
exp_ind = sys.argv[1]
re_num = int(sys.argv[2])
base_dir = sys.argv[3]

with open(f'{base_dir}/data/train_data/additional_train_image_ids_{exp_ind}.json', 'r') as fp:
    re_image_ids = json.load(fp)
random.shuffle(re_image_ids)
chunk_size = len(re_image_ids) // re_num
for i in range(re_num):
    start = i*chunk_size
    end = min(len(re_image_ids), (i+1)*chunk_size)
    chunk = re_image_ids[start:end]
    with open(f'{base_dir}/data/train_data/additional_train_image_ids_{exp_ind}_{i}.json', 'w') as fp:
        fp.write(json.dumps(chunk))
