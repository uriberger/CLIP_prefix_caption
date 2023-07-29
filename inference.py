from predict import Predictor
import json
import time
from flickr30k_utils import get_caption_data, get_test_ids, get_cn_caption_data, get_multi30k_test_data
from pascal_utils import get_caption_data as get_pascal_caption_data
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--gpt2_model', default='gpt2')
    parser.add_argument('--dataset')
    parser.add_argument('--split')
    parser.add_argument('--image_dir')
    parser.add_argument('--json_file')
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    model_name = 'coco'
    model_path = args.model_path
    gpt_type = args.gpt2_model
    output_file_name = args.output_file + '.json'

    predictor = Predictor()
    print('Setting up predictor...', flush=True)
    predictor.setup(model_name=model_name, model_path=model_path, gpt_type=gpt_type)
    print('Predictor set up!', flush=True)

    if args.json_file is not None:
        if args.dataset == 'COCO':
            image_id_to_split = {}
            with open('dataset_coco.json', 'r') as fp:
                coco_data = json.load(fp)['images']
                for sample in coco_data:
                    image_id = sample['cocoid']
                    image_id_to_split[image_id] = sample['filepath'].split('2014')[0]
        image_ids_file_path = args.json_file
        with open(image_ids_file_path, 'r') as fp:
            image_ids = json.load(fp)
        dataset = {}
        for image_id in image_ids:
            if args.dataset == 'COCO':
                split = image_id_to_split[image_id]
                image_dir_path = f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014'
                file_name = f'COCO_{split}2014_{str(image_id).zfill(12)}.jpg'
            elif args.dataset == 'flickr30k':
                image_dir_path = '/cs/labs/oabend/uriber/datasets/flickr30/images'
                file_name = f'{image_id}.jpg'
            dataset[image_id] = os.path.join(image_dir_path, file_name)
    elif args.dataset == 'COCO':
        assert args.split is not None, 'Please specify a split'
        # COCO Karpathy split
        with open('dataset_coco.json', 'r') as fp:
            data = json.load(fp)

        dataset = {}
        for sample in data['images']:
            if not sample['split'] == args.split:
                continue
            image_path = '/cs/labs/oabend/uriber/datasets/COCO/' + sample['filepath'] + '/' + sample['filename']
            image_id = sample['cocoid']
            dataset[image_id] = image_path
    elif args.dataset == 'flickr30k':
        assert args.split is not None, 'Please specify a split'
        # Flickr30k
        with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
            data = json.load(fp)

        dataset = {}
        for sample in data['images']:
            if not sample['split'] == args.split:
                continue
            image_path = f'/cs/labs/oabend/uriber/datasets/flickr30/images/{sample["filename"]}'
            image_id = int(sample['filename'].split('.jpg')[0])
            dataset[image_id] = image_path
    elif args.dataset == 'pascal':
        caption_data = get_pascal_caption_data()
        dataset = {}
        for sample in caption_data:
            dataset[sample['image_id']] = sample['image_path']
    elif args.dataset == 'flickr8kcn':
        # Flickr8kcn
        caption_data = get_cn_caption_data()
        dataset = {}
        for sample in caption_data:
            if sample['image_id'] in dataset:
                continue
            image_path = '/cs/labs/oabend/uriber/datasets/flickr30/images/' + str(sample['image_id']) + '.jpg'
            image_id = sample['image_id']
            dataset[image_id] = image_path
    elif args.dataset == 'multi30k':
        assert args.split is not None, 'Please specify a split'
        # Multi30k
        with open(f'multi30k_{args.split}_data.json', 'r') as fp:
            caption_data = json.load(fp)
        dataset = {}
        for sample in caption_data:
            if sample['image_id'] in dataset:
                continue
            image_path = '/cs/labs/oabend/uriber/datasets/flickr30/images/' + str(sample['image_id']) + '.jpg'
            image_id = sample['image_id']
            dataset[image_id] = image_path
    elif args.image_dir is not None:
        image_dir_path = args.image_dir
        dataset = {}
        for file_name in os.listdir(image_dir_path):
            image_id = int(file_name.split('_')[0])
            dataset[image_id] = os.path.join(image_dir_path, file_name)
    else:
        assert False

    print('Dataset size: ' + str(len(dataset)))
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
            with open(output_file_name, 'w') as fp:
                json.dump(res, fp)
        count += 1
        generated_caption = predictor.predict(image=image_path, model=model_name, use_beam_search=True)
        res.append({'image_id': image_id, 'caption': generated_caption, 'id': count})

    with open(output_file_name,  'w') as fp:
        json.dump(res, fp)

    print('Finished!')
