from pathlib import Path
import csv
import shutil

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================
# CHIP_SIZE = 256
# STRIDE = 128

# Stricter than before
MIN_VALID_FRACTION = 0.95
MIN_BOUNDARY_PIXELS = 30
MIN_POSITIVE_PIXELS = 200
MIN_NEGATIVE_PIXELS = 200

# CHIP_SIZE = 768
# STRIDE = 384
# MIN_VALID_FRACTION = 0.70
# MIN_BOUNDARY_PIXELS = 50
# MIN_POSITIVE_PIXELS = 1000
# MIN_NEGATIVE_PIXELS = 1000

# ------------------------------------------------------------
# EDIT THESE PATHS
# ------------------------------------------------------------
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

OUT_ROOT = Path("hed_data")
TRAIN_IMAGE_DIR = OUT_ROOT / "train" / "images"
TRAIN_MASK_DIR = OUT_ROOT / "train" / "masks"
VAL_IMAGE_DIR = OUT_ROOT / "val" / "images"
VAL_MASK_DIR = OUT_ROOT / "val" / "masks"
MANIFEST_PATH = OUT_ROOT / "chip_manifest.csv"


# ============================================================
# SETUP
# ============================================================
def reset_output_dirs():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for d in [TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def check_alignment(img_ds, mask_ds):
    assert img_ds.crs == mask_ds.crs, "CRS mismatch"
    assert img_ds.transform == mask_ds.transform, "Transform mismatch"
    assert img_ds.width == mask_ds.width, "Width mismatch"
    assert img_ds.height == mask_ds.height, "Height mismatch"


def get_valid_pixel_mask(img_chip: np.ndarray) -> np.ndarray:
    """
    Valid pixel = all bands finite and at least one band > 0.
    Shape returned: (H, W), dtype=bool
    """
    finite = np.all(np.isfinite(img_chip), axis=0)
    nonzero = np.any(img_chip > 0, axis=0)
    return finite & nonzero


def rgb_from_multiband(img_chip: np.ndarray) -> np.ndarray:
    """
    Assumes image bands are [B2, B3, B4, B8].
    Returns uint8 RGB using B4,B3,B2.
    """
    if img_chip.shape[0] < 3:
        raise ValueError("Expected at least 3 bands in image chip.")

    b2 = img_chip[0]
    b3 = img_chip[1]
    b4 = img_chip[2]

    rgb = np.stack([b4, b3, b2], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 0.3)
    rgb = rgb / 0.3
    rgb = (rgb * 255).round().astype(np.uint8)
    return rgb


def mask_to_uint8(mask_chip: np.ndarray) -> np.ndarray:
    return ((mask_chip > 0).astype(np.uint8) * 255)


def has_enough_valid_data(valid_mask: np.ndarray) -> bool:
    return valid_mask.mean() >= MIN_VALID_FRACTION


def touches_nodata_edge(valid_mask: np.ndarray) -> bool:
    border = np.concatenate([
        valid_mask[0, :],
        valid_mask[-1, :],
        valid_mask[:, 0],
        valid_mask[:, -1]
    ])
    return not np.all(border)


def apply_image_valid_mask_to_label(mask_chip: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Set label to 0 wherever the image is invalid.
    """
    mask_bin = (mask_chip > 0).astype(np.uint8)
    mask_bin[~valid_mask] = 0
    return mask_bin


def has_useful_mask(mask_chip: np.ndarray, valid_mask: np.ndarray) -> bool:
    """
    Evaluate usefulness only within valid image area.
    Require:
    - enough positive pixels
    - enough negative pixels
    - enough coastline boundary complexity
    """
    valid_pixels = valid_mask.sum()
    if valid_pixels == 0:
        return False

    mask_valid = mask_chip[valid_mask]

    pos = int((mask_valid > 0).sum())
    neg = int((mask_valid == 0).sum())

    if pos < MIN_POSITIVE_PIXELS:
        return False
    if neg < MIN_NEGATIVE_PIXELS:
        return False

    # Boundary complexity on full chip after invalid area is zeroed out
    diff_x = np.abs(np.diff(mask_chip, axis=1)).sum()
    diff_y = np.abs(np.diff(mask_chip, axis=0)).sum()
    boundary_pixels = int(diff_x + diff_y)

    return boundary_pixels >= MIN_BOUNDARY_PIXELS


def save_png(array: np.ndarray, path: Path):
    Image.fromarray(array).save(path)


# ============================================================
# MAIN CHIPPER
# ============================================================
def process_aoi(aoi_name: str, image_path: str, mask_path: str, split: str, manifest_rows: list):
    if split == "train":
        out_img_dir = TRAIN_IMAGE_DIR
        out_mask_dir = TRAIN_MASK_DIR
    elif split == "val":
        out_img_dir = VAL_IMAGE_DIR
        out_mask_dir = VAL_MASK_DIR
    else:
        raise ValueError(f"Unknown split: {split}")

    with rasterio.open(image_path) as img_ds, rasterio.open(mask_path) as mask_ds:
        check_alignment(img_ds, mask_ds)

        chip_id = 0
        rejected_invalid = 0
        rejected_nodata_edge = 0
        rejected_mask = 0

        for top in range(0, img_ds.height - CHIP_SIZE + 1, STRIDE):
            for left in range(0, img_ds.width - CHIP_SIZE + 1, STRIDE):
                window = Window(left, top, CHIP_SIZE, CHIP_SIZE)

                img_chip = img_ds.read(window=window).astype(np.float32)
                raw_mask_chip = mask_ds.read(1, window=window).astype(np.uint8)

                valid_mask = get_valid_pixel_mask(img_chip)

                if not has_enough_valid_data(valid_mask):
                    rejected_invalid += 1
                    continue

                if touches_nodata_edge(valid_mask):
                    rejected_nodata_edge += 1
                    continue

                mask_chip = apply_image_valid_mask_to_label(raw_mask_chip, valid_mask)

                if not has_useful_mask(mask_chip, valid_mask):
                    rejected_mask += 1
                    continue

                rgb_chip = rgb_from_multiband(img_chip)
                mask_png = mask_to_uint8(mask_chip)

                stem = f"{aoi_name}_{chip_id:05d}"
                img_out = out_img_dir / f"{stem}.png"
                mask_out = out_mask_dir / f"{stem}.png"

                save_png(rgb_chip, img_out)
                save_png(mask_png, mask_out)

                manifest_rows.append({
                    "stem": stem,
                    "aoi": aoi_name,
                    "split": split,
                    "left": left,
                    "top": top,
                    "chip_size": CHIP_SIZE,
                    "stride": STRIDE,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "valid_fraction": float(valid_mask.mean()),
                    "positive_pixels": int((mask_chip[valid_mask] > 0).sum()),
                    "negative_pixels": int((mask_chip[valid_mask] == 0).sum()),
                })

                chip_id += 1

        print(f"{aoi_name}: saved {chip_id} chips to split='{split}'")
        print(
            f"  rejected_invalid={rejected_invalid}, "
            f"rejected_nodata_edge={rejected_nodata_edge}, "
            f"rejected_mask={rejected_mask}"
        )


if __name__ == "__main__":
    reset_output_dirs()
    manifest_rows = []

    for aoi_name, cfg in AOI_CONFIG.items():
        process_aoi(
            aoi_name=aoi_name,
            image_path=cfg["image"],
            mask_path=cfg["mask"],
            split=cfg["split"],
            manifest_rows=manifest_rows,
        )

    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem", "aoi", "split", "left", "top",
                "chip_size", "stride", "image_path", "mask_path",
                "valid_fraction", "positive_pixels", "negative_pixels"
            ]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDone. Chips written to:")
    print(OUT_ROOT.resolve())
    print("Manifest saved to:")
    print(MANIFEST_PATH.resolve())