from Levenshtein import distance
import json
import sys

assert len(sys.argv) == 2
exp_ind = sys.argv[1]

with open(f'reformulation_experiment/data/infer/base_infer_on_train_{exp_ind}_en.json', 'r') as fp:
    orig_data = json.load(fp)

sample_num = len(orig_data)

with open(f'reformulation_experiment/data/infer/base_infer_on_train_{exp_ind}_en_reformulated.json', 'r') as fp:
    re_data = json.load(fp)

assert sample_num == len(re_data)
dists = [distance(orig_data[i]['caption'].split(), re_data[i]['caption'].split()) for i in range(sample_num)]
far_inds = [i for i in range(sample_num) if dists[i] > 25]
far_inds_dict = {i: True for i in far_inds}

with open(f'reformulation_experiment/data/translated_data/coco_translated_data_{exp_ind}.json', 'r') as fp:
    t_data = json.load(fp)
assert len(t_data) == sample_num

with open(f'reformulation_experiment/data/infer/base_infer_on_train_{exp_ind}_reformulated.json', 'r') as fp:
    re_de_data = json.load(fp)
assert len(re_de_data) == sample_num

new_data = [t_data[i] if i in far_inds_dict else re_de_data[i] for i in range(sample_num)]
with open(f'reformulation_experiment/data/re_train_data/reformulations_train_data_{exp_ind}.json', 'w') as fp:
    fp.write(json.dumps(new_data))
