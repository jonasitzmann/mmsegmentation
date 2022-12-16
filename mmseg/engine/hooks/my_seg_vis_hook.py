# Copyright (c) OpenMMLab. All rights reserved.
import torch
import os.path
import os.path as osp
import warnings
from typing import Sequence

import mmcv
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


@HOOKS.register_module()
class MySegVisHook(Hook):
    def __init__(self,
                 draw: bool = True,
                 graph=True,
                 num_images: int = 1,
                 wait_time: float = 0.,
                 file_client_args: dict = dict(backend='disk')):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.graph = graph
        self.num_images = num_images
        self.wait_time = wait_time
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.draw = draw

    def before_train(self, runner):
        runner.visualizer.add_graph(runner.model, None)

    def _after_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: dict,
        outputs: Sequence[SegDataSample],
        mode: str = 'val'
    ) -> None:
        if mode == 'train':
            wandb = runner.visualizer.get_backend('WandbVisBackend')
            if wandb:
                wandb.experiment.log(outputs)
                if runner.epoch == 0 and batch_idx == 0:
                    torch.onnx.export(runner.model, data_batch['inputs'][0], 'model.onnx')
                    wandb.save('model.onnx')
                    print('saved model')
            return
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)
        if batch_idx <= self.num_images:
            output = outputs[0]
            img_path = output.img_path
            img_bytes = self.file_client.get(img_path)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            window_name = f'{mode}_{osp.basename(img_path)}'
            if True: # using wandb
                classes = self._visualizer.dataset_meta.get('classes', None)
                wandb = runner.visualizer.get_backend('WandbVisBackend').experiment
                class_dict = {i: c for i, c in enumerate(classes)}
                log_img = wandb.Image(img, masks=dict(
                    predictions=dict(mask_data=output.pred_sem_seg.cpu().data[0].numpy(), class_labels=class_dict),
                    ground_truth=dict(mask_data=output.gt_sem_seg.cpu().data[0].numpy(), class_labels=class_dict),
                ))
                wandb.log({os.path.basename(img_path): log_img})
            # self._visualizer.add_datasample(
            #     window_name,
            #     img,
            #     data_sample=output,
            #     show=False,
            #     wait_time=self.wait_time,
            #     step=runner.iter)
