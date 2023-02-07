import argparse
import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
import csv

def create_database_from_csv(csv_file: str, caption_source: str):
    database = []
    with open('dataset_coco.json', 'r') as f:
        data = json.load(f)['images']
    print("%0d images loaded from json " % len(data))
    caption_count = 0

    with open(csv_file, 'r') as fp:
        my_reader = csv.reader(fp)
        first = True
        for row in my_reader:
            if first:
                first = False
                continue
            filename = row[1]
            image_id = int(filename.split('2014_')[1].split('.jpg')[0])
            dirname = row[0]
            filepath = f"/cs/labs/oabend/uriber/datasets/COCO/{dirname}/{filename}"
            assert os.path.isfile(filepath), 'Error: missing file in path ' + filepath
        
            res = {}
            res['image_id'] = image_id
            res['id'] = caption_count
            res['image_path'] = filepath
            if caption_source == 'gt':
                res['caption'] = row[6]
            elif caption_source == 'revised':
                res['caption'] = row[4]
            database.append(res)
            caption_count += 1

    return database

def create_database_from_image_ids(image_ids_file):
    database = []
    with open(image_ids_file, 'r') as fp:
        image_id_dict = {int(x.strip()): True for x in fp}
    with open('dataset_coco.json', 'r') as f:
        data = json.load(f)['images']
    for sample in data:
        image_id = sample['cocoid']
        if image_id not in image_id_dict:
            continue
        dirname = sample['filepath']
        image_id_full_str = str(image_id).zfill(12)
        filename = f'COCO_{dirname}_{image_id_full_str}.jpg'
        filepath = f"/cs/labs/oabend/uriber/datasets/COCO/{dirname}/{filename}"
        assert os.path.isfile(filepath), 'Error: missing file in path ' + filepath
    
        for sentence in sample['sentences']:
            res = {}
            res['image_id'] = image_id
            res['id'] = sentence['sentid']
            res['image_path'] = filepath
            res['caption'] = sentence['raw']
            database.append(res)

    return database

def main(clip_model_type: str, database: list, output_file: str):
    device = torch.device('cuda:0')
    out_path = f"./data/coco/{output_file}.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    print("%0d samples loaded" % len(database))
    all_embeddings = []
    all_captions = []
    caption_count = 0

    for sample in database:
        filepath = sample['image_path']
        image = io.imread(filepath)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        res = sample
        res["clip_embedding"] = caption_count
        all_embeddings.append(prefix)
        all_captions.append(res)
        caption_count += 1
        if (caption_count + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--caption_source', type=str, default='gt', choices=('gt', 'revised'))
    parser.add_argument('--image_ids_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='data')
    args = parser.parse_args()

    assert args.csv_file is not None or args.image_ids_file is not None, 'Error: please provide --csv_file or --image_ids_file'
    if args.csv_file is not None:
        database = create_database_from_csv(args.csv_file, args.caption_source)
    else:
        database = create_database_from_image_ids(args.image_ids_file)

    exit(main(args.clip_model_type, database, args.output_file))
