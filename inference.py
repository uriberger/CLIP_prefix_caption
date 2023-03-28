from predict import Predictor
import json
import time
import sys
from flickr30k_utils import get_caption_data, get_test_ids
import os

model_name = 'coco'
if len(sys.argv) == 2:
    model_path = 'coco_weights.pt'
else:
    model_path = sys.argv[2]

predictor = Predictor()
print('Setting up predictor...', flush=True)
predictor.setup(model_name=model_name, model_path=model_path)
print('Predictor set up!', flush=True)

if sys.argv[1] == 'COCO':
    # COCO Karpathy split
    with open('dataset_coco.json', 'r') as fp:
        data = json.load(fp)

    dataset = {}
    for sample in data['images']:
        if not sample['split'] == 'test':
            continue
        image_path = '/cs/labs/oabend/uriber/datasets/COCO/' + sample['filepath'] + '/' + sample['filename']
        image_id = sample['cocoid']
        dataset[image_id] = image_path
    res_name = 'COCO'
elif sys.argv[1] == 'flickr30k':
    # Flickr30k
    test_ids = get_test_ids()
    test_ids = {x: True for x in test_ids}
    caption_data = get_caption_data()
    dataset = {}
    for sample in caption_data:
        if not sample['image_id'] in test_ids:
            continue
        if sample['image_id'] in dataset:
            continue
        image_path = '/cs/labs/oabend/uriber/datasets/flickr30/images/' + str(sample['image_id']) + '.jpg'
        image_id = sample['image_id']
        dataset[image_id] = image_path
    res_name = 'flickr30k'
elif sys.argv[1].split('/')[-1].startswith('COCO_'):
    image_dir_path = sys.argv[1]
    dataset = {}
    for file_name in os.listdir(image_dir_path):
        image_id = int(file_name.split('_')[0])
        dataset[image_id] = os.path.join(image_dir_path, file_name)
    res_name = image_dir_path.split('/')[-1]
    
print('Generating captions...')
res = []
count = 0
checkpoint_len = 100
prev_checkpoint = time.time()
for image_id, image_path in dataset.items():
    if count % checkpoint_len == 0:
        time_from_prev = time.time() - prev_checkpoint
        prev_checkpoint = time.time()
        print('\tStarting sample ' + str(count) + ' out of ' + str(len(dataset)) + ', time from prev ' + str(time_from_prev), flush=True)
        with open('res_' + res_name + '.json', 'w') as fp:
            json.dump(res, fp)
    count += 1
    generated_caption = predictor.predict(image=image_path, model=model_name, use_beam_search=True)
    res.append({'image_id': image_id, 'caption': generated_caption, 'id': count})

with open('res_' + res_name + '.json',  'w') as fp:
    json.dump(res, fp)

print('Finished!')
    
