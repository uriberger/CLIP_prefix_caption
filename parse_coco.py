import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import csv


def main(clip_model_type: str, csv_file: str, caption_source: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/new_data_{clip_model_name}_train_{caption_source}.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('dataset_coco.json', 'r') as f:
        data = json.load(f)['images']
    print("%0d images loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
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
            image = io.imread(filepath)
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
            res = {}
            res["clip_embedding"] = caption_count
            res['image_id'] = image_id
            res['id'] = caption_count
            res['image_path'] = filepath
            if caption_source == 'gt':
                res['caption'] = row[6]
            elif caption_source == 'revised':
                res['caption'] = row[4]
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
    parser.add_argument('--csv_file', type=str)
    parser.add_argument('--caption_source', choices=('gt', 'revised'))
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.csv_file, args.caption_source))
