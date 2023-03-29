# Efficient Segmentation Comparison

## Branches

- mmseg: miou-fps
- mmengine:  v3_2_modified

## Extract Results from MMSEg

- MMSeg provides benchmarks (fps, mIoU) for various methods, model sizes, datasets, image resolutions.
- e.g. [https://github.com/open-mmlab/mmsegmentation/blob/1.x/configs/convnext/convnext.yml](https://github.com/open-mmlab/mmsegmentation/blob/1.x/configs/convnext/convnext.yml)
- Use `scripts/create_csvs.py` to summarize all results into a csv tables/from_mmseg/all.csv
    - each csv contains the following information for all experiments:
        - **name** of the model including the size, e.g. “Segformer_MIT-B0”
        - **method**, i.e. “segformer”
        - ****config** path for reproducing the results
        - **weights** url for reproducing the inference
        - **fps** measured by mmseg
        - **resolution** of the input images during inference
        - **hardware** used to measure fps (always V100)
        - **dataset**
        - **mIoU**
    - the methods are then grouped by dataset and resoluton to separte files
    tables/from_mmseg/<dataset>/<resolution>.csv

## Reproduce inference on own hardware

- The folder “scripts” contains several scripts to resimulate the inference on own hardware (e.g. gpus from our cluster or the local cpu)
- `do_inference.py` evaluates mIoU, fps and #parameters of a given configuration file.
    - the results are saved in `tables/reproduced<_prefix>/<config>.py`
    - the config should be contained in `tables/from_mmseg/all.csv` to load pre-trained weights
    - the parameter and flops count is based on `mmcv.cnn.get_model_complexity_info` but is highly unreliable! For most architectures, some flops are not counted. Check how each author is calculating the flops in the original repo (if present)!
    - if you are interested in only one of mIoU, flops or fps, comment out the corresponding lines
    - `do_inference.sh` and `do_inference_a100.sh` are wrappers of this script to submit a slurm job using a gtx 1080ti or A100 gpu respectively.
- `scripts/get_checkpoint.py` can be used to download the checkpoint of a config file listed in `tables/from_mmseg/all.csv`
- `reproduce_method.py` can be used to resimulate inference for all models (i.e. different model sizes) of a given method for a given dataset and image resolution.
    - the script then finds all relevant configs in `tables/from_mmseg/all.csv`
    - depending on the arguments, the script directly starts inference of these models (sequentially) or submits slurm jobs.
    - Also it can be used to only download the weights of these models using the `scripts/get_checkpoints.py` script.

## Plot the results

- `scripts/plot_method`  reads the generated csvs (both, from mmseg and reproduced) and visualizes the fps/mIoU trade-off for a given dataset and image resolution
    - each model is represented by a marker
    - models from the same method (e.g. “Segformer_MIT-B0” and “Segformer_MIT-B1”) are connected by lines
    - based on the flags `draw_repreoduced` and `draw_reported`  also the reproduced results are drawn.
        - they are connected to the original mmseg results by dotted lines
        - they shoud only differ in fps, not mIoU (since only the hardware, not the model weights are different). So, dotted lines should always be horizontal.
    - the attribute `is_interesting` determines, if ta method gets color and a legend entry. The others are plotted only for reference (half-transparent, gray, no legend entry). Most often, I used one of these criteria:
        - manual list of interesting methods. E.g. segformer and segnext
        - method x is interesting, if at least one model of x achieves the best performance for a given fps. Meaning there is no other model with a higher mIoU and fps.
        - method x is interesting if it has already been reproduced.
    
    ## Reproduce Trainings
    
    - Be aware of the batch size. E.g. the config `segformer_mit-b0_8xb2-160k_ade20k-512x512.py` uses only a batch size of 2 because it is made for a training with 8 GPUs (8xb2). When only using one gpu, increase the batch size to 8x2 = 16 to get the same preformance.
        - The largest models cannot be trained on a single A100 40GB with bs=16.  bs=12 had similar results.
    - So far I was able to reproduce trainings for segfomer but not for segnext on ADE20K. Maybe Cityscapes leads to more stable results?