import sys
import os
import json
import time
from bert_score import score
import statistics
from collections import defaultdict

lang = sys.argv[1]
gt_file = sys.argv[2]
input_patterns = sys.argv[3:]

with open(gt_file, 'r') as fp:
    gt_data = json.load(fp)['annotations']
image_id_to_captions = defaultdict(list)
for x in gt_data:
    image_id_to_captions[x['image_id']].append(x['caption'])
ref_list = list(image_id_to_captions.items())
ref_list = sorted(ref_list, key=lambda x:x[0])
references = [x[1] for x in ref_list]

outer_time = time.time()
for pattern in input_patterns:
    print(f'Starting {pattern}, time from prev {time.time()-outer_time}', flush=True)
    outer_time = time.time()
    
    file_name = pattern.split('/')[-1]
    assert '@' in file_name
    dir_path = '/'.join(pattern.split('/')[:-1])
    file_parts = file_name.split('@')
    assert len(file_parts) == 3
    options = file_parts[1].split(',')
    file_names = [file_parts[0] + x + file_parts[2] for x in options]
    file_paths = [os.path.join(dir_path, x) for x in file_names]
    
    precisions = []
    recalls = []
    f1s = []
    inner_time = time.time()
    for file_path in file_paths:
        print(f'\tStarting {file_path}, time from prev {time.time()-inner_time}', flush=True)
        inner_time = time.time()
        
        with open(file_path, 'r') as fp:
            cur_data = json.load(fp)
            cur_data = sorted(cur_data, key=lambda x:x['image_id'])
            predictions = [x['caption'] for x in cur_data]
            P, R, F1 = score(predictions, references, lang=lang)
            precisions.append(statistics.mean(P.tolist()))
            recalls.append(statistics.mean(R.tolist()))
            f1s.append(statistics.mean(F1.tolist()))
    print('---------------------')
    print(pattern)
    if len(precisions) > 1:
        print(f'Precision: {round(statistics.mean(precisions), 3)}+-{round(statistics.stdev(precisions), 3)}')
        print(f'Recall: {round(statistics.mean(recalls), 3)}+-{round(statistics.stdev(recalls), 3)}')
        print(f'F1: {round(statistics.mean(f1s), 3)}+-{round(statistics.stdev(f1s), 3)}', flush=True)
    else:
        print(f'Precision: {round(precisions[0], 3)}')
        print(f'Recall: {round(recalls, 3)}')
        print(f'F1: {round(f1s, 3)}', flush=True)
    print('---------------------')
