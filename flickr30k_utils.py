import os
import gzip

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

def get_cn_caption_data():
    root_dir_path = '/cs/labs/oabend/uriber/datasets/flickr8kcn'
    caption_dir_name = 'data'
    caption_file_name = 'flickr8kzhc.caption.txt'
    caption_file_path = os.path.join(root_dir_path, caption_dir_name, caption_file_name)
    
    image_id_captions_pairs = []
    with open(caption_file_path, 'r', encoding='utf8') as fp:
        for line in fp:
            striped_line = line.strip()
            if len(striped_line) == 0:
                continue
            line_parts = striped_line.split()
            assert len(line_parts) == 2
            image_id = int(line_parts[0].split('_')[0])
            caption = line_parts[1]
            image_id_captions_pairs.append({'image_id': image_id, 'caption': caption})
            
    return image_id_captions_pairs

def get_multi30k_test_data():
    root_dir_path = '/cs/labs/oabend/uriber/datasets/multi30k'
    data_dir_name = 'data'
    task_dir_name = 'task2'
    raw_dir_name = 'raw'
    image_splits_dir_name = 'image_splits'
        
    train_file_name_prefix = 'train'
    val_file_name_prefix = 'val'
    test_file_name_prefix = 'test_2016'
                    
    caption_file_name_suffix = 'de.gz'
    image_file_name_suffix = '_images.txt'
                            
    caption_inds = range(1, 6)
                            
    test_caption_file_names = [f'{test_file_name_prefix}.{i}.{caption_file_name_suffix}'
                               for i in caption_inds]
                                
    test_image_file_name = f'{test_file_name_prefix}{image_file_name_suffix}'
                                
    task_dir_path = os.path.join(root_dir_path, data_dir_name, task_dir_name)
                                
    caption_dir_path = os.path.join(task_dir_path, raw_dir_name)
    test_caption_file_paths = [os.path.join(caption_dir_path, test_caption_file_names[i])
                               for i in range(len(caption_inds))]
                                    
    image_dir_path = os.path.join(task_dir_path, image_splits_dir_name)
    test_image_file_path = os.path.join(image_dir_path, test_image_file_name)

    line_to_image_id = []
    image_file_path = test_image_file_path
    with open(image_file_path, 'r') as fp:
        for image_file_name in fp:
            image_id = int(image_file_name.split('.')[0])
            line_to_image_id.append(image_id)

    image_id_captions_pairs = []
    caption_file_paths = test_caption_file_paths
    for caption_file_path in caption_file_paths:
        with gzip.open(caption_file_path, 'r') as fp:
            line_ind = 0
            for line in fp:
                caption = line.strip().decode('utf-8')
                image_id_captions_pairs.append({'image_id': line_to_image_id[line_ind], 'caption': caption})
                line_ind += 1

    return image_id_captions_pairs
                                    
def get_test_ids():
    with open('flickr30k_test.txt', 'r') as fp:
        res = [int(x.strip()) for x in fp]
    return res
