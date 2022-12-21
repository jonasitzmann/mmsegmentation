import pandas as pd
import os
from my_utils.fix_wget import fix_wget
import wget
fix_wget()


def get_checkpoint(config_path):
    ckpt_file = config_path.replace('configs/', 'downloaded_ckpts/') + '.pth'
    if not os.path.exists(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        df = pd.read_csv('tables/from_mmseg/all.csv')
        ckpt_url = df[df.config==config_path].weights.iloc[0]
        ckpt_url = ckpt_url.replace('https:', 'http:')
        print(f'downloading weights from {ckpt_url}')
        wget.download(ckpt_url, ckpt_file)
    return ckpt_file


if __name__ == '__main__':
    ckpt_file = get_checkpoint('configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py')
    print(ckpt_file)