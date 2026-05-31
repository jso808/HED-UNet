# HED-UNet for Greenland Coastline Delineation

This repository adapts **HED-UNet** for automated coastline delineation in western Greenland. HED-UNet is a boundary-aware deep learning model that jointly learns semantic segmentation and edge detection, making it useful for extracting sharp land-water boundaries from remote sensing imagery.

The original HED-UNet model was developed for Antarctic coastline and glacier-front monitoring using Sentinel-1 SAR imagery. In this project, the model has been adapted to use **Sentinel-2 optical imagery** and Danish Geodata Agency / GST coastline-derived masks for Greenland coastline segmentation.

## Project Overview

This implementation trains HED-UNet to classify pixels as land or water and to learn coastline edges using paired image and mask chips. The workflow is:

1. Prepare aligned Sentinel-2 image rasters and binary land-water masks.
2. Generate training and validation chips using `make_hed_chips.py`.
3. Train HED-UNet using `train.py` and `config.yml`.
4. Run full-AOI inference using `infer_full_aoi_mask.py`.
5. Export probability and binary mask rasters for downstream coastline vectorization and evaluation.

## Adaptation from Original HED-UNet

The original HED-UNet repository included examples for Antarctic coastline monitoring and building footprint extraction. This project modifies the workflow for Greenland coastline delineation.

Main changes include:

- Uses **Sentinel-2 optical imagery** instead of the original Sentinel-1 SAR setup.
- Uses three RGB input channels derived from Sentinel-2 bands:
  - B4 = Red
  - B3 = Green
  - B2 = Blue
- Sentinel-2 B8 / NIR is present in the source imagery but is not used as a model input because the current HED-UNet configuration expects three channels.
- Uses custom Greenland AOI image and mask pairs.
- Uses a custom chip-generation script to create HED-UNet-compatible PNG image and mask folders.
- Uses a held-out AOI validation strategy:
  - AOI 2 and AOI 3 are used for training.
  - AOI 1 is used for validation.
- Adds full-raster tiled inference for applying the trained model to a complete AOI GeoTIFF.

## Repository Files

### `make_hed_chips.py`

Generates image and mask chips from aligned AOI GeoTIFFs.

The script:

- Reads Sentinel-2 image rasters and aligned binary mask rasters.
- Checks that image and mask CRS, transform, width, and height match.
- Extracts sliding-window chips.
- Filters chips based on:
  - valid-data fraction,
  - no-data edges,
  - minimum land/water pixel counts,
  - minimum coastline boundary complexity.
- Converts Sentinel-2 bands `[B2, B3, B4, B8]` into RGB PNG chips using `[B4, B3, B2]`.
- Writes output to:

```text
hed_data/
  train/
    images/
    masks/
  val/
    images/
    masks/
  chip_manifest.csv

```

Example AOI split:

```python
AOI_CONFIG = {
    "AOI_1": {
        "image": "AOI_1_BEST_20170723.tif",
        "mask": "AOI_1_mask_aligned.tif",
        "split": "val",
    },
    "AOI_2": {
        "image": "AOI_2_BEST.tif",
        "mask": "AOI_2_mask_aligned.tif",
        "split": "train",
    },
    "AOI_3": {
        "image": "AOI_3_BEST.tif",
        "mask": "AOI_3_mask_aligned.tif",
        "split": "train",
    },
}
```

Run with:

```bash
python make_hed_chips.py
```

## Training

Training is controlled by `train.py` and `config.yml`.

The model configuration is:

```yaml
model: HEDUNet
model_args:
  input_channels: 3
  base_channels: 16
  stack_height: 5
  batch_norm: true

feature_pyramid: true

loss_args:
  type: AutoBCE

batch_size: 8
epochs: 20
learning_rate: 0.001

data_threads: 4

visualization_tiles: [0, 7, 14, 38, 120, 74, 57, 39, 101]
```

The training script expects the chip folder structure created by `make_hed_chips.py`:

```text
hed_data/train/images
hed_data/train/masks
hed_data/val/images
hed_data/val/masks
```

Run training with:

```bash
python train.py --config=config.yml
```

To resume from a checkpoint:

```bash
python train.py --config=config.yml --resume=logs/<run_name>/checkpoints/<epoch>.pt
```

To print a model summary:

```bash
python train.py --summary --config=config.yml
```

Training outputs are saved to:

```text
logs/
  YYYY-MM-DD_HH-MM-SS/
    config.yml
    metrics.txt
    checkpoints/
    figures/
```

## Full-AOI Inference

Use `infer_full_aoi_mask.py` to apply a trained checkpoint to a complete AOI raster.

The inference script:

- Loads the trained HED-UNet checkpoint.
- Reads the full Sentinel-2 AOI GeoTIFF.
- Converts `[B2, B3, B4, B8]` input into RGB `[B4, B3, B2]`.
- Normalizes inputs to match training.
- Runs tiled inference using overlapping windows.
- Averages overlapping predictions.
- Saves:
  - a probability raster,
  - a thresholded binary mask raster.

Key settings:

```python
CONFIG_PATH = "config.yml"
CHECKPOINT_PATH = "logs/2026-04-09_05-18-13/checkpoints/09.pt"

INPUT_IMAGE_PATH = "AOI_1_full_2017.tif"

OUTPUT_PROB_PATH = "AOI_1_pred_prob.tif"
OUTPUT_MASK_PATH = "AOI_1_pred_mask.tif"

CHIP_SIZE = 256
STRIDE = 128
BATCH_SIZE = 8
THRESHOLD = 0.5
```

Run inference with:

```bash
python infer_full_aoi_mask.py
```

The probability raster stores continuous model confidence values. The binary mask raster applies a threshold of `0.5` to classify pixels as land or water.

## Data Requirements

Input imagery should be aligned GeoTIFFs with Sentinel-2 bands ordered as:

```text
B2, B3, B4, B8
```

The current model uses only RGB bands:

```text
B4, B3, B2
```

The mask raster must:

- have the same CRS as the image,
- have the same transform,
- have the same width and height,
- contain binary land-water labels.

## Important Notes

The current implementation uses only three input channels. Although Sentinel-2 NIR / B8 is available in the exported imagery, it is not currently passed into the model. This is a limitation because NIR is often useful for separating water, snow, ice, and land. To use B8 directly, the model would need to be retrained with `input_channels: 4`, and the chip-generation, data-loading, and inference scripts would need to preserve four-channel inputs rather than converting to RGB PNGs.

The workflow is designed for Greenland coastline extraction and may not generalize directly to other Arctic or non-Arctic regions without additional training data and validation.

## Citation

This project is adapted from HED-UNet. If using the original model architecture or codebase, cite:

```tex
@article{HEDUNet2021,
  author={Heidler, Konrad and Mou, Lichao and Baumhoer, Celia and Dietz, Andreas and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HED-UNet: Combined Segmentation and Edge Detection for Monitoring the Antarctic Coastline}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2021.3064606}
}
```

## Acknowledgment

This repository builds on the original HED-UNet implementation and adapts it for a Greenland coastline delineation research project using Sentinel-2 imagery and GST-derived coastline reference masks.
