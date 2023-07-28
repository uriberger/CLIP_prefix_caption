import torch
import clip
from PIL import Image
import json
import sys
import os
import time
import statistics

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset_name = sys.argv[1]
if dataset_name == 'COCO':
    image_id_to_path = lambda image_id: '/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
elif dataset_name == 'pascal':
    images_root = '/cs/labs/oabend/uriber/datasets/pascal_sentences/dataset'
    image_id_to_path_dict = {}
    for dir_name in os.listdir(images_root):
        dir_path = os.path.join(images_root, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for file_name in os.listdir(dir_path):
            image_id = int(file_name.split('2008_')[1].split('.jpg')[0])
            image_id_to_path_dict[image_id] = os.path.join(dir_path, file_name)
    image_id_to_path = lambda image_id: image_id_to_path_dict[image_id]
else:
    assert False, 'Unknown dataset ' + dataset_name

input_patterns = sys.argv[2:]
data = {}
for pattern in input_patterns:
    file_name = pattern.split('/')[-1]
    if '@' in file_name:
        dir_path = '/'.join(pattern.split('/')[:-1])
        file_parts = file_name.split('@')
        assert len(file_parts) == 3
        options = file_parts[1].split(',')
        file_names = [file_parts[0] + x + file_parts[2] for x in options]
        file_paths = [os.path.join(dir_path, x) for x in file_names]
        single_file = False
    else:
        file_paths = [pattern]
        single_file = True
    data[pattern] = {}
    for file_path in file_paths:
        with open(file_path, 'r') as fp:
            data[pattern][file_path] = json.load(fp)
            data[pattern][file_path] = sorted(data[pattern][file_path], key=lambda x:x['image_id'])

ind_to_file = []
for pattern, file_list in data.items():
    for file_name in file_list.keys():
        ind_to_file.append((pattern, file_name))

res = {}
for pattern, file_list in data.items():
    res[pattern] = {}
    for file_name in file_list.keys():
        res[pattern][file_name] = 0

first_file_data = data[ind_to_file[0][0]][ind_to_file[0][1]]
sample_num = len(first_file_data)

t = time.time()
for i in range(sample_num):
    if i % 1000 == 0:
        print('Staring sample ' + str(i) + ' out of ' + str(sample_num) + ', time from prev ' + str(time.time() - t), flush=True)
        t = time.time()
    image_id = first_file_data[i]['image_id']
    for pattern, file_name in ind_to_file:
        assert image_id == data[pattern][file_name][i]['image_id']
    image_path = image_id_to_path(image_id)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    caption_list = []
    for pattern, file_name in ind_to_file:
        caption_list.append(data[pattern][file_name][i]['caption'])
    text = clip.tokenize(caption_list, truncate=True).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        for j in range(probs.shape[0]):
            pattern, file_name = ind_to_file[j]
            res[pattern][file_name] += probs[j]

for pattern, pattern_res in res.items():
    if single_file:
        print(f'{pattern}: {round((list(pattern_res.values())[0])/sample_num, 3)}')
    else:
        mean = round(statistics.mean([x/sample_num for x in pattern_res.values()]), 3)
        stdev = round(statistics.stdev([x/sample_num for x in pattern_res.values()]), 3)
        print(f'{pattern}: {mean} +- {stdev}')
