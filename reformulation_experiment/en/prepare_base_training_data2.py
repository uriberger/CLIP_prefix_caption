import json
import os

res_file = 'reformulation_experiment/en/data/base_train_data/flickr_train_data.json'
if not os.path.isfile(ref_file):
    with open('/cs/labs/oabend/uriber/datasets/flickr30k/karpathy/dataset_flickr30k.json', 'r') as fp:
        flickr_data = json.load(fp)['images']
    flickr_train_data = [x for x in flickr_data if x['split'] == 'train']

    res = [{'image_id': int(x['filename'].split('.jpg')[0]), 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x["filename"]}', 'sentences': x['sentences']} for x in flickr_train_data]

    with open(res_file, 'w') as fp:
        fp.write(json.dumps(res))
