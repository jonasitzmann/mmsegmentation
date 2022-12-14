import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from tools.analysis_tools.benchmark import benchmark
from tools.analysis_tools.get_flops import get_model_complexity_info

from mmseg.utils import register_all_modules

def do_inference(config_path, ckpt=None, input_shape=(3, 512, 512)):
    register_all_modules()
    cfg = Config.fromfile(config_path)
    cfg['test_dataloader']['dataset']['indices'] = 10
    cfg.work_dir = 'inference_results'
    cfg.load_from = ckpt
    runner = Runner.from_cfg(cfg)
    runner.model.eval()
    flops, params = get_model_complexity_info(runner.model, input_shape)
    benchmark_metrics = benchmark(config_path, repeat_times=1, n_images=10)
    metrics = runner.test()



if __name__ == '__main__':
    do_inference('configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py')
