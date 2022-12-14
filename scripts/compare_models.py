import yaml
from glob import glob
import pandas as pd
from dotdict import dotdict
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import os
import numpy as np

def main():
    yaml_files = glob('configs/**/*.yml')
    res_list = []
    for f in list(yaml_files):
        res_list += extract_results(f)
    df = pd.DataFrame([dict(r) for r in res_list])
    df = df.sort_values(by='fps', ascending=False)
    filtered = df[(df['dataset'] == 'ADE20K') & (df['resolution'] == '(512,512)')]
    # filtered = df[(df['dataset'] == 'ADE20K') & (df['resolution'] == '(640,640)')]
    # filtered = df[(df['dataset'] == 'Cityscapes') & (df['resolution'] == '(512,1024)')]
    filtered['is_best'] = False
    best_mIoU = 0
    for i, mIoU in enumerate(filtered.mIoU):
        if mIoU > best_mIoU:
            filtered['is_best'].iloc[i] = True
            best_mIoU = mIoU

    fig, ax = plt.subplots()
    for method, sub_df in filtered.groupby('method'):
        plot_kw = dotdict()
        if any(sub_df.is_best):
        # if method == 'hrnet':
            plot_kw.label = method
        else:
            plot_kw.color = 'gray'
        ax.plot(sub_df.fps, sub_df.mIoU, **plot_kw)
    plt.legend()
    plt.show()


def extract_results(yaml_file):
    res_list = []
    with open(yaml_file, 'r') as f:
        file_content = yaml.safe_load(f)
    dir_name = os.path.basename(os.path.dirname(yaml_file))
    models = file_content['Models']
    for model in models:
        try:
            d_model = dotdict()
            d_model.name = model['In Collection']
            d_model.method = dir_name
            mdat = model['Metadata']
            if 'backbone' in mdat:
                d_model.name += '_' + mdat['backbone']
            d_model.config = model['Config']
            d_model.weights = model['Weights']
            d_inf_time = mdat['inference time (ms/im)']
            assert len(d_inf_time) == 1, 'd_inf_time has multiple elements'
            d_inf_time = d_inf_time[0]
            d_model.fps = 1000 / d_inf_time['value']
            d_model.resolution = d_inf_time['resolution']
            d_model.hardware = d_inf_time['hardware']
            results = model['Results']
            results = [r for r in results if r['Task'] == 'Semantic Segmentation']
            assert len(results) == 1, 'd_inf_time has multiple elements'
            res = results[0]
            d_res = dotdict(d_model)
            d_res.dataset = res['Dataset']
            d_res.mIoU = res['Metrics']['mIoU']
            res_list.append(d_res)
        except KeyError as key_error:
            pass
            # print(key_error)
    return res_list



if __name__ == '__main__':
    main()