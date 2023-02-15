import torch
import numpy as np
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import PixelData
from mmseg.utils import OptSampleList
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def binary_encode_labels(labels: PixelData, num_bits_encode, generator_matrix=None):
    encoded = labels.clone()
    h, w = encoded.shape
    data = encoded.data
    mask = 2**torch.arange(num_bits_encode - 1, -1, -1).to(data.device, data.dtype)
    mask = mask.view(-1, 1, 1).repeat_interleave(h, 1).repeat_interleave(w, 2)
    enc_data = data.bitwise_and(mask).ne(0).int()
    encoded.set_data(dict(data=enc_data))
    return encoded


def add_binary_encoding(data_samples: OptSampleList, num_bits):
    assert data_samples is not None, 'data_samples should not be None for binary encoding of labels'
    for data_sample in data_samples:
        gt = data_sample.gt_sem_seg
        data_sample.set_data(dict(gt_encoded=binary_encode_labels(gt, num_bits)))


def decode_logits(logits, num_classes, generator_matrix=None):
    b, num_bits, h, w = logits.shape
    mask = 2**torch.arange(num_bits-1, -1, -1).to(logits.device, torch.int)
    mask = mask[None, :, None, None].expand_as(logits)
    binary = (logits > 0)
    decoded = (mask * binary).sum(1).clamp(0, num_classes - 1).to(torch.long)
    decoded = torch.nn.functional.one_hot(decoded, num_classes=num_classes)
    decoded = decoded.permute(0, 3, 1, 2).to(logits.dtype)
    return decoded


def get_encoding_matrix(num_classes):
    num_bits = int(np.ceil(np.log2(num_classes)))
    mask = 2**torch.arange(num_bits-1, -1, -1).view(1, -1).expand(num_classes, -1)
    classes = torch.arange(0, num_classes).view(-1, 1).expand(-1, num_bits)
    m = (torch.bitwise_and(mask, classes) > 0).int()
    m = m.permute(1, 0)
    return m, num_bits


def get_encoding_conv(num_classes):
    m, num_bits = get_encoding_matrix(num_classes)
    m = m[:, :, None, None].to(torch.float32)
    conv = torch.nn.Conv2d(num_classes, num_bits, kernel_size=1)
    conv.weight.data = m
    conv.bias.data = torch.zeros_like(conv.weight[:, 0, 0, 0]) - 0.5
    return conv


def plot_enc_layer(output, output_enc):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    output = output.detach().cpu().numpy()
    output_enc = output_enc.detach().cpu().numpy()
    sample = output[0, :, 0, 0]
    sample_enc = output_enc[0, :, 0, 0]
    ax1.bar(height=sample, x=np.arange(sample.shape[0]))
    ax2.bar(height=sample_enc, x=np.arange(sample_enc.shape[0]))
    fig.suptitle(f'argmax={np.argmax(sample)}')
    plt.show()


def _test_encoding_matrix():
    num_classes = 8
    conv = get_encoding_conv()
    gt = torch.arange(0, num_classes).view(1, -1)
    onehot = torch.nn.functional.one_hot(gt, num_classes=num_classes).permute(2, 0, 1).to(torch.float32)
    result = conv(onehot).permute(1, 2, 0)
    print(f'{result=}')


def _test_binary_encode_labels():
    register_all_modules()
    cfg = Config.fromfile('hamming_code/hamming_test_cfg.py')
    dataloader = Runner.build_dataloader(cfg.train_dataloader)
    data_preprocessor = MODELS.build(cfg.data_preprocessor)
    batch = data_preprocessor(next(iter(dataloader)), training=True)
    for data_sample in batch['data_samples']:
        gt = data_sample.gt_sem_seg
        data_sample.set_data(dict(gt_encoded=binary_encode_labels(gt, num_bits_encode=8)))
    print(batch)

if __name__ == '__main__':
    # binary_encode_labels(PixelData(data=torch.arange(0, 4).view(1, 2, 2)))
    # _test_binary_encode_labels()
    _test_encoding_matrix()