import os.path

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
from dotdict import dotdict
from glob import glob
import numpy as np


def main():
    csv_dir = 'tables/from_mmseg'
    dataset_name = 'ADE20K'
    resolution = '(512,512)'
    df = pd.read_csv(f'{csv_dir}/{dataset_name}/{resolution}.csv')
    df['config_name'] = df.config.apply(lambda x: os.path.basename(x) if x and isinstance(x, str) else x)
    df = find_best_methods(df)
    df_reproduced = read_reproduced_results()
    if df_reproduced is None:
        df_merged = df
    else:
        df_reproduced.columns = [col + '_reproduced' if not col=='config_name' else col for col in df_reproduced.columns]
        df_merged = pd.merge(df, df_reproduced, how='outer', on='config_name')
    plot_miou_over_fps(df_merged)


def read_reproduced_results():
    dfs = []
    filenames = glob('tables/reproduced/*.csv')
    if not filenames:
        return None
    for filename in filenames:
        config_name = os.path.basename(filename)[:-4]
        df = pd.read_csv(filename)
        df['config_name'] = config_name
        dfs.append(df)
    reproduced_df = pd.concat(dfs)
    return reproduced_df


def plot_miou_over_fps(df_mmseg):
    fig, ax = plt.subplots(dpi=200, figsize=(11, 6))
    # ax2 = ax.twiny()
    # ax2.set_xlabel('fps (reproduced)')
    for method, sub_df in df_mmseg.groupby('method'):
        is_reproduced = 'fps_reproduced' in sub_df.columns and len(sub_df.fps_reproduced.dropna()) > 0
        plot_kw = dotdict(marker='X', ms=4)
        if is_reproduced:
            plot_kw.zorder = 10
            plot_kw.lw = 2
            plot_kw.ms = 6
        else:
            plot_kw.lw = 1.5

        if any(sub_df.is_best):
            plot_kw.label = method
            plot_kw.alpha=0.6
        else:
            plot_kw.color = 'gray'
            plot_kw.alpha=0.3
        plot_kw.color = ax.plot(sub_df.fps, sub_df.mIoU, **{**plot_kw, **(dict(label=None)if is_reproduced else {})})[0]._color
        if is_reproduced:
            # if method == 'segformer':
            #     print(sub_df.params_reproduced / 1e6)
                plot_kw.marker='o'
                plot_kw.ms=8
                ms = sub_df.params_reproduced.apply(lambda x: x / 100_000)
                ms = sub_df.flops_reproduced.apply(lambda x: x / 1000_000_000)
                ax.plot(sub_df.fps_reproduced, sub_df.mIoU_reproduced, **{**plot_kw, 'alpha': 1, 'lw': 3,
                                                                          # 'ms': 0
                                                                          })
                # ax.scatter(sub_df.fps_reproduced, sub_df.mIoU_reproduced, s=ms)
                for i, row in sub_df.dropna(axis=0).iterrows():
                    plt.plot(
                        [row.fps, row.fps_reproduced],
                        [row.mIoU, row.mIoU_reproduced],
                        color=plot_kw.color,
                        ls='dotted',
                        alpha=plot_kw.alpha,
                        zorder=5
                    )


    ax.set_xlabel('fps')
    ax.set_ylabel('mIoU')
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_best_methods(df):
    df['is_best'] = False
    best_mIoU = 0
    for i, mIoU in enumerate(df.mIoU):
        if mIoU > best_mIoU:
            df['is_best'].iloc[i] = True
            best_mIoU = mIoU
    return df


if __name__ == '__main__':
    main()
