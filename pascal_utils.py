import os

def get_caption_data():
    root_dir_path = '/cs/labs/oabend/uriber/datasets/pascal_sentences'
    sentences_dir_path = os.path.join(root_dir_path, 'sentence')
    
    caption_data = []
    subdir_names = sorted(os.listdir(sentences_dir_path))
    for subdir_name in subdir_names:
        subdir_path = os.path.join(sentences_dir_path, subdir_name)
        if os.path.isdir(subdir_path):
            file_names = sorted(os.listdir(subdir_path))
            for file_name in file_names:
                image_id = int(file_name.split('_')[1].split('.txt')[0])
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path, 'r') as fp:
                    for line in fp:
                        caption = line.strip()
                        caption_data.append({'image_id': image_id, 'caption': caption, 'image_path': os.path.join(subdir_path, file_name.replace('.txt', '.jpg')).replace('/sentence/', '/dataset/')})

    return caption_data
