import argparse
import os
import json
import pandas as pd

def postprocess(results_dir):
    log_file_path = os.path.join(results_dir, 'train_log.txt')
    with open(log_file_path, 'r') as fp:
        result_list = [json.loads(x.strip().replace('\'', '"')) for x in fp if 'Bleu' in x]
    column_names = list(result_list[0].keys())

    res_dict = {x: ['{:.5f}'.format(float(y[x])) for y in result_list] for x in column_names}
    df = pd.DataFrame(res_dict)
    df.to_csv('res.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None, required=True)

    args = parser.parse_args()

    postprocess(args.results_dir)
