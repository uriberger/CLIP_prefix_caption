import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str, image_ids_file: str):
    with open(image_ids_file, 'r') as fp:
        image_ids_dict = {int(x.strip()): True for x in fp}
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('dataset_coco.json', 'r') as f:
        data = json.load(f)['images']
    print("%0d images loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    image_count = 0
    caption_count = 0
    for image_count in tqdm(range(len(data))):
        d = data[image_count]
        img_id = d["cocoid"]
        if img_id not in image_ids_dict:
            continue
        dirname = d['filepath']
        filename = d['filename']
        filepath = f"/cs/labs/oabend/uriber/datasets/COCO/{dirname}/{filename}"
        assert os.path.isfile(filepath), 'Error: missing file in path ' + filepath
        image = io.imread(filepath)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        for sentence in d['sentences']:
            res = {}
            res["clip_embedding"] = caption_count
            res['image_id'] = img_id
            res['id'] = sentence['sentid']
            res['caption'] = sentence['raw']
            all_embeddings.append(prefix)
            all_captions.append(res)
            caption_count += 1
        if (image_count + 1) % 10000 == 0:
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
    parser.add_argument('--image_ids_file', type=str)
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.image_ids_file))
