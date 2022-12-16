import yaml
from glob import glob
import pandas as pd
from dotdict import dotdict
import os


def main():
    csv_dir = 'tables/from_mmseg'
    groups = create_csvs(csv_dir)


def create_csvs(csv_dir):
    yaml_files = glob('configs/**/*.yml')
    res_list = []
    for f in list(yaml_files):
        res_list += extract_results(f)
    df = pd.DataFrame([dict(r) for r in res_list])
    df = df.sort_values(by='fps', ascending=False)
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(f'{csv_dir}/all.csv', index=False)
    groups = {k: v for k, v in df.groupby(['dataset', 'resolution'])}
    for (ds, res), group in groups.items():
        ds_dir = f'{csv_dir}/{ds}'
        os.makedirs(ds_dir, exist_ok=True)
        group.to_csv(f'{csv_dir}/{ds}/{res}.csv', index=False)
    return groups


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