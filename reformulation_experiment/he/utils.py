import json
import os

def get_cross_model_data():
    cross_modal_root_path = '/cs/labs/oabend/uriber/datasets/crossmodal3600'
    with open(os.path.join(cross_modal_root_path, 'captions.jsonl'), 'r') as fp:
        json_list = list(fp)
    data = [json.loads(x) for x in json_list]
    res = []
    for sample in data:
        image_id = int(sample['image/key'], 16)
        image_path = os.path.join(cross_modal_root_path, 'images', f'{sample["image/key"]}.jpg')
        sentences = [{'raw': x} for x in sample['he']['caption']]
        res.append('image_id': image_id, 'image_path': image_path, 'sentences': sentences)
    return res
