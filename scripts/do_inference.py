import argparse
import os
import os.path as osp
import pandas as pd
import torch.cuda

from mmengine.config import Config
from mmengine.runner import Runner
from tools.analysis_tools.benchmark import benchmark
from tools.analysis_tools.get_flops import get_model_complexity_info

from mmseg.utils import register_all_modules
from scripts.get_checkpoint import get_checkpoint
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    do_inference(args.config)


def do_inference(config_path, use_ckpt=True, input_shape=(3, 512, 512)):
    debug = (not torch.cuda.is_available()) or (os.environ.get('MACHINE', default='cluster') != 'cluster')
    print(f'{debug=}')
    repeat_times, n_images_benchmark, n_warmup, n_images_total = (1, 4, 2, 5) if debug else (3, 200, 5, None)
    ckpt = get_checkpoint(config_path) if use_ckpt else None
    register_all_modules()
    cfg = Config.fromfile(config_path)
    if n_images_total:  # if not specified use entire dataset
        cfg['test_dataloader']['dataset']['indices'] = n_images_total
    cfg['visualizer']['vis_backends'][0]['init_kwargs']['project'] = 'mmseg_inference'
    cfg.work_dir = 'inference_results'
    cfg.load_from = ckpt
    runner = Runner.from_cfg(cfg)
    runner.model.eval()
    flops, params = get_model_complexity_info(runner.model, input_shape, as_strings=False)
    benchmark_metrics = benchmark(config_path, repeat_times=repeat_times, n_images=n_images_benchmark, n_warmup=n_warmup)
    metrics = runner.test()
    results = dict(
        flops=flops,
        params=params,
        fps=benchmark_metrics['average_fps'],
        fps_var=benchmark_metrics['fps_variance'],
        mIoU=metrics['mIoU']
    )
    save_dir = 'tables/reproduced'
    if debug:
        save_dir += '_debug'
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame([results])
    filename = f'{save_dir}/{osp.basename(config_path)}.csv'
    df.to_csv(filename, index=False)
    print(f'{filename=}')




if __name__ == '__main__':
    main()

