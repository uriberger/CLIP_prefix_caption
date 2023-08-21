import json
import sys

assert len(sys.argv) == 4
input_file = sys.argv[1]
output_file = sys.argv[2]
exp_ind = int(sys.argv[3])

with open(input_file, 'r') as fp:
    data = json.load(fp)

with open(f'reformulation_experiment/data/image_ids/val_image_ids_{exp_ind}.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

res = []
for sample in data:
    image_id = sample['image_id']
    if image_id in image_ids_dict:
        res.append({'image_id': image_id, 'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg', 'sentences': [{'raw': sample['caption']}]})
    
with open(output_file, 'w') as fp:
    fp.write(json.dumps(res))
