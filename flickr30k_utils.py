import os

def image_file_name_to_id(image_file_name):
    return int(image_file_name.split('.')[0])

def get_caption_data():
    root_dir_path = '/cs/labs/oabend/uriber/datasets/flickr30'
    tokens_dir_name = 'tokens'
    tokens_file_name = 'results_20130124.token'
    tokens_file_path = os.path.join(root_dir_path, tokens_dir_name, tokens_file_name)
    
    image_id_captions_pairs = []
    with open(tokens_file_path, encoding='utf-8') as fp:
        for line in fp:
            split_line = line.strip().split('g#')
            img_file_name = split_line[0] + 'g'
            image_id = image_file_name_to_id(img_file_name)
            sent_id = int(split_line[1].split('\t')[0])
            caption = split_line[1].split('\t')[1]  # The first token is caption number
            image_id_captions_pairs.append({'image_id': image_id, 'caption': caption, 'id': sent_id})

    return image_id_captions_pairs

def get_test_ids():
    with open('flickr30k_test.txt', 'r') as fp:
        res = [int(x.strip()) for x in fp]
    return res
