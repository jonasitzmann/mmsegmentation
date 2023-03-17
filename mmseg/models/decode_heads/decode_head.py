# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor
import numpy as np

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize
from hamming_code.encode import add_binary_encoding, decode_logits, get_encoding_conv, plot_enc_layer
from mmseg.models.losses.cross_entropy_loss import cross_entropy
import matplotlib.pyplot as plt


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg.
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg')),
                 encode_labels=False,
                 map_to_n_bits=None,
                 encoding_conv_manual_weights=False,
                 softmax_hardness=1,
                 ):
        super().__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.encode_labels = encode_labels
        self.num_classes = num_classes
        if self.encode_labels:
            if map_to_n_bits is None:
                num_classes = int(np.ceil(np.log2(num_classes)))
                out_channels = num_classes
            if loss_decode['type'] == 'CrossEntropyLoss':
                loss_decode.use_sigmoid = True
            self.gt_key = 'gt_encoded'
        else:
            self.gt_key = 'gt_sem_seg'

        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.out_channels = out_channels
        self.threshold = threshold




        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        self.binary_encoding_conv = None
        self.output_layer_name = 'conv_seg'
        if map_to_n_bits is not None:
            self.softmax_hardness = softmax_hardness
            self.output_layer_name = 'binary_encoding_conv'
            self.binary_encoding_conv = nn.Conv2d(self.out_channels, map_to_n_bits, kernel_size=1)
            if encoding_conv_manual_weights:
                encoding_conv_manual = get_encoding_conv(self.out_channels)
                self.binary_encoding_conv.weight.data = encoding_conv_manual.weight.data
                self.binary_encoding_conv.bias.data = encoding_conv_manual.bias.data
                self.binary_encoding_conv.requires_grad = True

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None


    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        if self.binary_encoding_conv is not None:
            output_onehot = torch.nn.functional.softmax(output * self.softmax_hardness)
            output_enc = self.binary_encoding_conv(output_onehot)
            # plot_enc_layer(output_onehot, output_enc)
            return output_enc
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        # print(f'{seg_logits.sum()=}')
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> List[Tensor]:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            List[Tensor]: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)
        if self.encode_labels:
            seg_logits_decoded = decode_logits(seg_logits, self.num_classes)
            return self.predict_by_feat(seg_logits_decoded, batch_img_metas)
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList, gt_key=None) -> Tensor:
        gt_key = gt_key or self.gt_key
        gt_semantic_segs = [
            data_sample.get(gt_key).data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            # mode='bilinear',
            # align_corners=self.align_corners,
            mode='nearest',
            align_corners=None,
        )
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if self.encode_labels:  # in this case ignore_idx has no effect, so seg_weight is set to ignore samples
            seg_label_decoded = self._stack_batch_gt(batch_data_samples, 'gt_sem_seg').squeeze(1)
            seg_weight = seg_label_decoded != self.ignore_index
            seg_weight = seg_weight.unsqueeze(1).expand_as(seg_logits)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index
                )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        # loss = dict(loss_ce=loss_decode(seg_logits, seg_label, ignore_index=self.ignore_index, weight=seg_weight))

        if self.encode_labels:
            loss['acc_bit'] = ((seg_logits > 0) == seg_label).sum() / seg_logits.numel()
            seg_logits_decoded = decode_logits(seg_logits, self.num_classes)
            loss['ce_decoded'] = cross_entropy(seg_logits_decoded, seg_label_decoded, ignore_index=self.ignore_index)
        #     valid_idxs = seg_weight[:, 0, :, :].nonzero()
        #     # seen_labels = []
        #     # for b, h, w in valid_idxs:
        #     b, h, w = valid_idxs[0]
        #     logits_sample = torch.tensor(seg_logits[b, :, h, w].detach(), requires_grad=True)
        #     # logits_sample = seg_logits[b, :, h, w]
        #     label_dec = torch.tensor(seg_label_decoded[b, h, w]).item()
        #     # if not label_dec in seen_labels:
        #     #     seen_labels.append(label_dec)
        #     # else:
        #     #     continue
        #     label_enc = seg_label[b, :, h, w]
        #     correct = seg_logits_decoded[b, :, h, w].argmax() == label_dec
        #     sample_loss = loss_decode(logits_sample[None, :, None, None], label_enc[None, :, None, None])
        #     sample_loss.backward()
        #     xs = np.arange(logits_sample.shape[0])
        #     ys = torch.sigmoid(logits_sample.detach().cpu())
        #     ys_2 = torch.sigmoid(logits_sample.detach().cpu() - 10 * logits_sample.grad.cpu())
        #     for i, (x, y1, y2) in enumerate(zip(xs, ys, ys_2)):
        #         plt.arrow(x, y1, 0, y2-y1, head_width=0.1, color='black', head_length=0.02, width=0.02, length_includes_head=True, label=None if i else 'neg. gradient')
        #     plt.scatter(xs, label_enc.cpu(), label='ground truth', alpha=1, s=100)
        #     plt.scatter(xs, ys, s=100, label='prediction')
        #     plt.axhline(0.5, 0, max(xs), color='gray')
        #     plt.ylim(0, 1)
        #     plt.suptitle(f'{sample_loss=:.4f}, {label_dec=}, {correct=}')
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.xticks(xs, (np.arange(len(xs))+1)[::-1])
        #     plt.show()
        #     self.zero_grad()

        else:
            seg_logits_decoded, seg_label_decoded = seg_logits, seg_label
        loss['acc_seg'] = accuracy(
            seg_logits_decoded, seg_label_decoded, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners,
            # mode='nearest',
            # align_corners=None
        )
        return seg_logits
