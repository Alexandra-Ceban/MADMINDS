from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label, sobel
from scipy.spatial.distance import cdist


IMAGE_DIR = Path(__file__).parent / "images-20261114T135605Z-1-001" / "images"
OUTPUT_PATH = Path(__file__).parent / "test-output.txt"
EXPECTED_COUNT = 28


def unique_images() -> list[Path]:
    """Drop byte-identical duplicate files, preferring names without '(1)'."""
    by_digest: dict[str, list[Path]] = defaultdict(list)
    for path in IMAGE_DIR.glob("*.png"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest[digest].append(path)
    result = []
    for copies in by_digest.values():
        result.append(min(copies, key=lambda p: ("(1)" in p.name, p.name)))
    return sorted(result, key=lambda p: int(p.stem.split("_")[1].split("(")[0]))


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def shell_features(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)
    hue, saturation, value = hsv.transpose(2, 0, 1)

    shell = (hue >= 45) & (hue <= 100) & (saturation >= 22) & (value <= 245)
    shell = binary_closing(shell, structure=np.ones((5, 5), dtype=bool))
    shell = binary_fill_holes(largest_component(shell))


    near_shell = binary_dilation(shell, iterations=5)
    rust = (
        near_shell
        & (hue >= 4)
        & (hue <= 43)
        & (saturation >= 35)
        & (value >= 20)
        & (value <= 185)
    )
    dark_pit = near_shell & (value <= 35)
    rust_labels, rust_count = label(binary_closing(rust, structure=np.ones((3, 3), bool)))
    rust_sizes = np.bincount(rust_labels.ravel())[1:] if rust_count else np.array([])
    rust_max_component = int(rust_sizes.max()) if len(rust_sizes) else 0

    ys, xs = np.where(shell)
    if len(xs) < 500:

        x0, x1, y0, y1 = 120, 390, 0, 512
    else:
        pad_x, pad_y = 8, 8
        x0, x1 = max(0, xs.min() - pad_x), min(512, xs.max() + pad_x + 1)
        y0, y1 = max(0, ys.min() - pad_y), min(512, ys.max() + pad_y + 1)

    rgb_crop = rgb[y0:y1, x0:x1]
    hsv_crop = hsv[y0:y1, x0:x1]
    mask_crop = shell[y0:y1, x0:x1].astype(np.uint8) * 255

    size = (32, 128)
    resized_rgb = np.asarray(
        Image.fromarray(rgb_crop).resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    resized_hsv = np.asarray(
        Image.fromarray(hsv_crop).resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    resized_mask = np.asarray(
        Image.fromarray(mask_crop).resize(size, Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0

    gray = resized_rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gx, gy = sobel(gray, axis=1), sobel(gray, axis=0)
    grad = np.hypot(gx, gy)

    features = []
    for row in range(8):
        for col in range(16):
            ys0, ys1 = row * 16, (row + 1) * 16
            xs0, xs1 = col * 2, (col + 1) * 2
            patch_mask = resized_mask[ys0:ys1, xs0:xs1] > 0.5
            features.append(float(patch_mask.mean()))
            if patch_mask.any():
                patch_hsv = resized_hsv[ys0:ys1, xs0:xs1][patch_mask]
                patch_rgb = resized_rgb[ys0:ys1, xs0:xs1][patch_mask]
                features.extend(patch_hsv.mean(axis=0).tolist())
                features.extend(patch_hsv.std(axis=0).tolist())
                features.extend(patch_rgb.std(axis=0).tolist())
                features.append(float(grad[ys0:ys1, xs0:xs1][patch_mask].mean()))
            else:
                features.extend([0.0] * 10)

    features.extend(
        [
            float(shell.mean()),
            float((x1 - x0) / max(1, y1 - y0)),
            float((xs.mean() / 512) if len(xs) else 0.5),
            float((ys.mean() / 512) if len(ys) else 0.5),
            float(rust.sum() / max(1, shell.sum())),
            float(dark_pit.sum() / max(1, shell.sum())),
            float(rust_max_component / max(1, shell.sum())),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def anomaly_scores(features: np.ndarray) -> np.ndarray:
    """Combine robust local-feature deviation and nearest-neighbor distance."""
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    scale = np.maximum(1.4826 * mad, 0.015)
    robust = np.clip((features - median) / scale, -12, 12)

    distances = cdist(robust, robust, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, 10, axis=1)[:, :10]
    neighbor_score = nearest.mean(axis=1)


    patch_width = 11
    patch_vectors = robust[:, : 8 * 16 * patch_width].reshape(-1, 8 * 16, patch_width)
    patch_scores = np.linalg.norm(patch_vectors, axis=2)
    local_score = np.sort(patch_scores, axis=1)[:, -8:].mean(axis=1)

    def percentile(values: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        return order.astype(np.float32) / max(1, len(values) - 1)

    return 0.5 * percentile(neighbor_score) + 0.5 * percentile(local_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()

    paths = unique_images()
    features = np.stack([shell_features(path) for path in paths])
    scores = anomaly_scores(features)
    ranking = np.argsort(scores)[::-1]
    candidates = [paths[index].name for index in ranking[:EXPECTED_COUNT]]

    print("Top rust/color streak feature values (review these first):")
    rust_rank = np.argsort(features[:, -1])[::-1][:12]
    for index in rust_rank:
        print(
            f"{paths[index].name}\t"
            f"rust={features[index, -3]:.5f}\t"
            f"dark={features[index, -2]:.5f}\t"
            f"component={features[index, -1]:.5f}"
        )

    print(f"Unique image contents: {len(paths)}")
    print(f"Scored features: {features.shape}")
    print("Top anomaly candidates:")
    for name, index in zip(candidates, ranking[:EXPECTED_COUNT]):
        print(f"{name}\t{scores[index]:.4f}")

    if args.write_output:
        OUTPUT_PATH.write_text("\n".join(candidates) + "\n", encoding="utf-8")
        print(f"Wrote {EXPECTED_COUNT} names to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
