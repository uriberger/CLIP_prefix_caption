import os
import gzip

def get_caption_data_for_split(split):
    image_dir_path = '/cs/labs/oabend/uriber/datasets/multi30k/data/task2/image_splits'
    
    line_to_image_id = []
    image_file_path = os.path.join(image_dir_path, f'{split}_images.txt')
    with open(image_file_path, 'r') as fp:
        for image_file_name in fp:
            image_id = int(image_file_name.split('.')[0])
            line_to_image_id.append(image_id)

    image_id_captions_pairs = []
    caption_inds = range(1, 6)
    caption_dir_path = '/cs/labs/oabend/uriber/datasets/multi30k/data/task2/raw'
    caption_file_names = [f'{split}.{i}.de.gz' for i in caption_inds]
    caption_file_paths = [os.path.join(caption_dir_path, caption_file_names[i]) for i in range(len(caption_inds))]
    for caption_file_path in caption_file_paths:
        with gzip.open(caption_file_path, 'r') as fp:
            line_ind = 0
            for line in fp:
                caption = line.strip().decode('utf-8')
                image_id_captions_pairs.append({'image_id': line_to_image_id[line_ind], 'caption': caption})
                line_ind += 1
                
    return image_id_captions_pairs
