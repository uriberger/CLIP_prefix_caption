import sys
import json

assert len(sys.argv) == 2
model_name = sys.argv[1]

with open(f'reformulation_experiment/de/data/translated_data/coco_{model_name}_de_translated_helsinki.json', 'r') as fp:
    coco_data = json.load(fp)
with open(f'reformulation_experiment/de/data/translated_data/aic_{model_name}_de_translated_helsinki.json', 'r') as fp:
    aic_data = json.load(fp)
data = coco_data + aic_data
with open(f'reformulation_experiment/de/data/translated_train_data/coco_aic_{model_name}_train_data.json', 'w') as fp:
    fp.write(json.dumps(data))
