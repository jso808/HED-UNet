from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import torch
import yaml

from deep_learning import get_model


# ============================================================
# USER SETTINGS
# ============================================================
CONFIG_PATH = "config.yml"
CHECKPOINT_PATH = "logs/2026-04-09_05-18-13/checkpoints/09.pt"

INPUT_IMAGE_PATH = "AOI_1_full_2017.tif"

# Outputs
OUTPUT_PROB_PATH = "AOI_1_pred_prob - subset - 3.tif"
OUTPUT_MASK_PATH = "AOI_1_pred_mask - subset - 3.tif"

# Inference tiling
CHIP_SIZE = 256
STRIDE = 128
BATCH_SIZE = 8

# Threshold for binary mask
THRESHOLD = 0.5


# ============================================================
# HELPERS
# ============================================================
def build_model(config_path: str, checkpoint_path: str, device: torch.device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    modelclass = get_model(config["model"])
    model = modelclass(**config["model_args"])
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, config


def rgb_from_multiband_float(img_chip: np.ndarray) -> np.ndarray:
    """
    Assumes image bands are [B2, B3, B4, B8].
    Returns float RGB in [0,1], using B4,B3,B2.
    Input shape: (bands, h, w)
    Output shape: (3, h, w)
    """
    if img_chip.shape[0] < 3:
        raise ValueError("Expected at least 3 bands in image chip.")

    b2 = img_chip[0]
    b3 = img_chip[1]
    b4 = img_chip[2]

    rgb = np.stack([b4, b3, b2], axis=0)  # (3, h, w)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 0.3)
    rgb = rgb / 0.3  # [0,1]
    return rgb.astype(np.float32)


def normalize_for_model(rgb_chip: np.ndarray) -> np.ndarray:
    """
    Convert [0,1] RGB to [-1,1], matching training.
    """
    return rgb_chip * 2.0 - 1.0


def valid_pixel_mask(img_chip: np.ndarray) -> np.ndarray:
    """
    Valid pixel = all bands finite and at least one band > 0.
    Returns shape (h, w), bool.
    """
    finite = np.all(np.isfinite(img_chip), axis=0)
    nonzero = np.any(img_chip > 0, axis=0)
    return finite & nonzero


def generate_windows(width: int, height: int, chip_size: int, stride: int):
    """
    Sliding windows that also ensure right/bottom edges are covered.
    """
    xs = list(range(0, max(width - chip_size + 1, 1), stride))
    ys = list(range(0, max(height - chip_size + 1, 1), stride))

    if len(xs) == 0 or xs[-1] != width - chip_size:
        xs.append(max(width - chip_size, 0))
    if len(ys) == 0 or ys[-1] != height - chip_size:
        ys.append(max(height - chip_size, 0))

    # unique + sorted
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for top in ys:
        for left in xs:
            yield Window(left, top, chip_size, chip_size), left, top


# ============================================================
# MAIN INFERENCE
# ============================================================
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")

    model, config = build_model(CONFIG_PATH, CHECKPOINT_PATH, device)

    with rasterio.open(INPUT_IMAGE_PATH) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()

        # Accumulators for overlap averaging
        prob_sum = np.zeros((height, width), dtype=np.float32)
        prob_count = np.zeros((height, width), dtype=np.float32)

        windows = list(generate_windows(width, height, CHIP_SIZE, STRIDE))
        print(f"Total windows: {len(windows)}")

        batch_tensors = []
        batch_meta = []

        for window, left, top in tqdm(windows):
            img_chip = src.read(window=window).astype(np.float32)  # (bands, h, w)

            # Skip if the chip is mostly invalid
            valid_mask = valid_pixel_mask(img_chip)
            if valid_mask.mean() == 0:
                continue

            rgb = rgb_from_multiband_float(img_chip)
            rgb = normalize_for_model(rgb)

            batch_tensors.append(torch.from_numpy(rgb))
            batch_meta.append((window, left, top, valid_mask))

            # Run batch
            if len(batch_tensors) == BATCH_SIZE:
                process_batch(model, batch_tensors, batch_meta, prob_sum, prob_count, device)
                batch_tensors = []
                batch_meta = []

        # Remainder
        if len(batch_tensors) > 0:
            process_batch(model, batch_tensors, batch_meta, prob_sum, prob_count, device)

        # Average overlapping predictions
        prob_avg = np.zeros_like(prob_sum, dtype=np.float32)
        valid_out = prob_count > 0
        prob_avg[valid_out] = prob_sum[valid_out] / prob_count[valid_out]

        # Binary mask
        pred_mask = (prob_avg >= THRESHOLD).astype(np.uint8)

        # Save probability raster
        prob_profile = profile.copy()
        prob_profile.update(
            count=1,
            dtype="float32",
            compress="lzw",
            nodata=0
        )

        with rasterio.open(OUTPUT_PROB_PATH, "w", **prob_profile) as dst:
            dst.write(prob_avg, 1)

        # Save binary mask raster
        mask_profile = profile.copy()
        mask_profile.update(
            count=1,
            dtype="uint8",
            compress="lzw",
            nodata=0
        )

        with rasterio.open(OUTPUT_MASK_PATH, "w", **mask_profile) as dst:
            dst.write(pred_mask, 1)

    print("Saved:")
    print(f"  Probability raster: {Path(OUTPUT_PROB_PATH).resolve()}")
    print(f"  Binary mask raster: {Path(OUTPUT_MASK_PATH).resolve()}")


@torch.no_grad()
def process_batch(model, batch_tensors, batch_meta, prob_sum, prob_count, device):
    batch = torch.stack(batch_tensors, dim=0).to(device)  # (B, 3, H, W)

    # Model returns final output and intermediate outputs
    y_hat, _ = model(batch)

    # Channel 0 = segmentation, channel 1 = edge
    seg_prob = torch.sigmoid(y_hat[:, 0]).cpu().numpy()  # (B, H, W)

    for i, (window, left, top, valid_mask) in enumerate(batch_meta):
        h = int(window.height)
        w = int(window.width)

        chip_prob = seg_prob[i]

        # Only accumulate where image was valid
        chip_prob = chip_prob * valid_mask.astype(np.float32)

        prob_sum[top:top+h, left:left+w] += chip_prob
        prob_count[top:top+h, left:left+w] += valid_mask.astype(np.float32)


if __name__ == "__main__":
    run_inference()