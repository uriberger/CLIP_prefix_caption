import json
import os
import sys

assert len(sys.argv) == 2
exp_ind = int(sys.argv[1])

output_file = 'reformulation_experiment/zh/data/base_train_data/train_data.json'
if not os.path.isfile(output_file):
    with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r') as fp:
        aic_train_data = json.load(fp)

    res = [{'image_id': int(x['image_id'].split('.jpg')[0], 16), 'image_path': f'/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_images_20170902/{x["image_id"]}', 'sentences': [{'raw': y} for y in x['caption']]} for x in aic_train_data]

    with open(output_file, 'w') as fp:
        fp.write(json.dumps(res))
