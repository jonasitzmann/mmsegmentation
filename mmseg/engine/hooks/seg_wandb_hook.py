# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.evaluator
import wandb

from mmseg.registry import HOOKS
import numpy as np
from mmengine.runner import Runner
from mmengine.dist import master_only
from mmengine.hooks import CheckpointHook
from mmengine.hooks.logger_hook import LoggerHook
import os
from torchvision.transforms.functional import to_pil_image
from mmengine import MessageHub
import torch
# from mmengine.hooks import DistEvalHook, EvalHook


@HOOKS.register_module()
class MMSegWandbHook(LoggerHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_evaluator: mmengine.evaluator.Evaluator = None
        self.wandb = None
        self.ckpt_hook = None
        self.val_dataset = None
        self.num_eval_images = 2
        self.last_ckpt_path = None

    @master_only
    def before_run(self, runner: Runner):
        super().before_run(runner)
        wandb_backend = runner.visualizer.get_backend('WandbVisBackend')
        self.wandb = wandb_backend.experiment if wandb_backend is not None else None
        # Check if EvalHook and CheckpointHook are available.
        # Check conditions to log evaluation
        if runner.val_evaluator is None:
            runner.logger.warning('val evaluator is required')
        else:
            self.val_dataset = runner.val_dataloader.dataset
            # Determine the number of samples to be logged.
            if self.num_eval_images > len(self.val_dataset):
                self.num_eval_images = len(self.val_dataset)
                runner.logger.warning(
                    f'{self.num_eval_images=}) > {len(self.val_dataset)=}. All val data will be logged.')

        # self._init_data_table()
        # self._add_ground_truth(runner)
        # self._log_data_table()

    def log_graph(self, runner, data_batch):
        try:
            batch = runner.model.data_preprocessor(data_batch, training=True)
            filename = f'{runner.log_dir}/model.onnx'
            torch.onnx.export(runner.model, batch['inputs'], filename)
            self.wandb.save(filename)
            print('saved model graph')
        except Exception:
            print('could not save graph')

    def after_train_iter(self, runner, batch_idx, data_batch, outputs, *args, **kwargs):
        if batch_idx == 0 and runner.epoch==0:
            self.log_graph(runner, data_batch)
        arg_dict = dict(runner=runner, batch_idx=batch_idx, data_batch=data_batch, outputs=outputs)
        super_results = super().after_train_iter(*args, **arg_dict, **kwargs)
        return super_results
        # return super_results if runner.model.training else self._after_train_iter(*+arg_dict)

    def after_val_iter(self, runner, batch_idx, data_batch, outputs, *args, **kwargs):
        arg_dict = dict(runner=runner, batch_idx=batch_idx, data_batch=data_batch, outputs=outputs)
        super().after_val_iter(*args, **arg_dict, **kwargs)
        self._after_val_iter(**arg_dict)

    def after_val_epoch(self, runner, metrics, *args, **kwargs):
        super().after_val_epoch(runner, metrics)
        last_ckpt_path = MessageHub.get_current_instance()._runtime_info.get('best_ckpt')
        if last_ckpt_path != self.last_ckpt_path:
            self.last_ckpt_path = last_ckpt_path
            print(f'saving checkpoint {last_ckpt_path}')

    @master_only
    def _after_val_iter(self, runner: Runner, batch_idx, data_batch, outputs):
        return
        # Save checkpoint and metadata
        if self.log_checkpoint: # when do this?
            if self.log_checkpoint_metadata and self.val_evaluator:
                metadata = {
                    'iter': runner.iter + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'iter_{runner.iter+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir, f'iter_{runner.iter+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation:
            results = runner.val_evaluator.evalutate(1) # todo: dont compute results here
            # Initialize evaluation table
            self._init_pred_table()
            # Log predictions
            self._log_predictions(results, runner)
            # Log the table
            self._log_eval_table(runner.iter + 1)

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = wandb.Artifact(
            f'run_{self.wandb.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.val_evaluator.evaluate(1)
        results = self.val_evaluator.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.val_evaluator.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['image_name', 'ground_truth', 'prediction']
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self, runner):
        # Get image loading pipeline
        from mmcv.transforms import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        if img_loader is None:
            self.log_evaluation = False
            runner.logger.warning('LoadImageFromFile is required to add images to W&B Tables.')
            return

        self.eval_image_indexs = np.arange(len(self.val_dataset))
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]
        classes = self.val_dataset.METAINFO['classes']
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}
        self.class_set = self.wandb.Classes([{'id': id, 'name': name} for id, name in self.class_id_to_label.items()])

        for idx in self.eval_image_indexs:
            x = self.val_dataset[idx]
            inputs, data_samples = x['inputs'], x['data_samples']
            image_name = os.path.basename(data_samples.img_path)
            image = to_pil_image(inputs.data[0])
            seg_mask = data_samples.gt_sem_seg.data[0]
            if seg_mask.ndim == 2:
                wandb_masks = {
                    'ground_truth': {
                        'mask_data': seg_mask.numpy(),
                        'class_labels': self.class_id_to_label
                    }
                }
                self.data_table.add_data(
                    image_name,
                    self.wandb.Image(image, masks=wandb_masks, classes=self.class_set))
            else:
                runner.logger.warning(f'The segmentation mask is {seg_mask.ndim}D which is not supported by W&B.')
                return

    def _log_predictions(self, results, runner):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)
        assert len(results) == len(self.val_dataset)

        for i, eval_image_index in enumerate(self.eval_image_indexs):
            # Get the result
            pred_mask = results[eval_image_index]

            if pred_mask.ndim == 2:
                wandb_masks = {
                    'prediction': {
                        'mask_data': pred_mask,
                        'class_labels': self.class_id_to_label
                    }
                }

                # Log a row to the data table.
                self.eval_table.add_data(
                    self.data_table_ref.data[i][0],
                    self.data_table_ref.data[i][1],
                    self.wandb.Image(
                        self.data_table_ref.data[i][1],
                        masks=wandb_masks,
                        classes=self.class_set))
            else:
                runner.logger.warning(
                    'The prediction segmentation mask is '
                    f'{pred_mask.ndim}D which is not supported by W&B.')
                self.log_evaluation = False
                return

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val_dataset', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self, iter):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        self.wandb.log_artifact(pred_artifact)
