"""Monte Carlo unsupervised anomaly ranking for the Laos shell images.

The baseline is the dominant, internally consistent group of shells. OpenCV
extracts aligned color, form, texture, and surface features; bootstrap Gaussian
models then estimate how far each shell falls into the baseline's tails.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_DIR = Path(__file__).parent / "images-20261114T135605Z-1-001" / "images"
OUTPUT_PATH = Path(__file__).parent / "test-output.txt"
EXPECTED_COUNT = 28


def unique_images() -> list[Path]:
    """Remove byte-identical copies, preferring names without '(1)'."""
    by_digest: dict[str, list[Path]] = defaultdict(list)
    for path in IMAGE_DIR.glob("*.png"):
        by_digest[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)

    unique = [
        min(copies, key=lambda path: ("(1)" in path.name, path.name))
        for copies in by_digest.values()
    ]
    return sorted(unique, key=lambda path: int(path.stem.split("_")[1].split("(")[0]))


def segment_shell(bgr: np.ndarray) -> np.ndarray:
    """Use HSV color and connected components to isolate the centered shell."""
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array([35, 30, 20], dtype=np.uint8),
        np.array([70, 255, 245], dtype=np.uint8),
    )
    green = cv2.morphologyEx(
        green, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        green, connectivity=8
    )
    mask = np.zeros_like(green)
    min_area = max(1500, int(height * width * 0.005))
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component]
        center_x = centroids[component][0]
        if (
            0.37 * width <= center_x <= 0.63 * width
            and box_width <= 0.30 * width
            and box_height >= 0.10 * height
            and area >= min_area
        ):
            mask[labels == component] = 255

    if cv2.countNonZero(mask) == 0 and count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask[labels == largest] = 255
    return mask


def fill_shell_outline(mask: np.ndarray) -> np.ndarray:
    """Bridge the white belt and fill the shell's row-wise silhouette."""
    height, width = mask.shape
    valid_rows = np.flatnonzero(np.count_nonzero(mask, axis=1) >= 8)
    silhouette = np.zeros_like(mask)
    if len(valid_rows) < 2:
        return mask.copy()

    left = np.full(height, np.nan, dtype=np.float32)
    right = np.full(height, np.nan, dtype=np.float32)
    for row in valid_rows:
        columns = np.flatnonzero(mask[row])
        left[row], right[row] = np.percentile(columns, [2, 98])

    rows = np.arange(valid_rows[0], valid_rows[-1] + 1)
    left_edges = np.rint(np.interp(rows, valid_rows, left[valid_rows])).astype(int)
    right_edges = np.rint(np.interp(rows, valid_rows, right[valid_rows])).astype(int)
    for row, x0, x1 in zip(rows, left_edges, right_edges):
        silhouette[row, max(0, x0 - 1) : min(width, x1 + 2)] = 255
    return silhouette


def align_shell(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the shell's main axis upright and keep its narrow tip at the top."""
    mask = segment_shell(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        points = np.vstack(contours).astype(np.float32)
        vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = math.degrees(math.atan2(float(vy), float(vx)))
        rotation = 90.0 - angle
        while rotation > 90.0:
            rotation -= 180.0
        while rotation < -90.0:
            rotation += 180.0

        height, width = mask.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1.0)
        bgr = cv2.warpAffine(
            bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    silhouette = fill_shell_outline(mask)
    widths = np.count_nonzero(silhouette, axis=1)
    occupied_rows = np.flatnonzero(widths > 0)
    if len(occupied_rows) >= 10:
        occupied_widths = widths[occupied_rows]
        end = max(1, len(occupied_widths) // 10)
        top_width = float(np.median(occupied_widths[:end]))
        bottom_width = float(np.median(occupied_widths[-end:]))
        if top_width > bottom_width * 1.05:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
            silhouette = cv2.rotate(silhouette, cv2.ROTATE_180)
    return bgr, silhouette


def histogram(values: np.ndarray, mask: np.ndarray, bins: int, high: float) -> list[float]:
    """Return a normalized OpenCV histogram for one channel."""
    hist = cv2.calcHist([values], [0], mask, [bins], [0, high]).flatten()
    hist /= max(1.0, float(hist.sum()))
    return hist.astype(np.float32).tolist()


def shell_features(path: Path) -> np.ndarray:
    """Describe overall shell appearance plus color/texture in local regions."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    bgr, silhouette = align_shell(bgr)

    ys, xs = np.where(silhouette > 0)
    image_height, image_width = silhouette.shape
    if len(xs) == 0:
        x0, x1, y0, y1 = image_width // 4, 3 * image_width // 4, 0, image_height
        silhouette = np.zeros_like(silhouette)
        silhouette[:, x0:x1] = 255
    else:
        x0, x1 = max(0, int(xs.min()) - 8), min(image_width, int(xs.max()) + 9)
        y0, y1 = max(0, int(ys.min()) - 8), min(image_height, int(ys.max()) + 9)

    rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    shell_mask = silhouette[y0:y1, x0:x1]
    target_size = (48, 192)
    rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    shell_mask = cv2.resize(shell_mask, target_size, interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(lab[:, :, 0])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    # Mark rust and dark pixels only inside the estimated shell silhouette.
    rust = cv2.bitwise_and(
        cv2.inRange(
            hsv,
            np.array([3, 35, 20], dtype=np.uint8),
            np.array([30, 255, 185], dtype=np.uint8),
        ),
        shell_mask,
    )
    dark = cv2.bitwise_and(
        cv2.inRange(
            hsv,
            np.array([0, 0, 0], dtype=np.uint8),
            np.array([179, 255, 35], dtype=np.uint8),
        ),
        shell_mask,
    )

    features: list[float] = []

    # Color histograms retain broad color distributions without background pixels.
    for channel, bins, limit in ((0, 8, 180), (1, 4, 256), (2, 4, 256)):
        features.extend(histogram(hsv[:, :, channel], shell_mask, bins, limit))

    # Global Lab color and edge statistics.
    shell_pixels = shell_mask > 0
    lab_pixels = lab[shell_pixels].astype(np.float32) / 255.0
    edge_pixels = gradient[shell_pixels]
    features.extend(lab_pixels.mean(axis=0).tolist())
    features.extend(lab_pixels.std(axis=0).tolist())
    edge_hist, _ = np.histogram(edge_pixels, bins=8, range=(0, 2), density=False)
    features.extend((edge_hist / max(1, edge_hist.sum())).astype(np.float32).tolist())
    features.extend([float(edge_pixels.mean()), float(edge_pixels.std())])

    # Local measurements: four length bands by three width bands.
    for row in range(4):
        y_start, y_end = row * 48, (row + 1) * 48
        for col in range(3):
            x_start, x_end = col * 16, (col + 1) * 16
            local_mask = shell_mask[y_start:y_end, x_start:x_end] > 0
            if local_mask.any():
                local_lab = lab[y_start:y_end, x_start:x_end][local_mask].astype(np.float32) / 255.0
                local_hsv = hsv[y_start:y_end, x_start:x_end][local_mask]
                local_gradient = gradient[y_start:y_end, x_start:x_end][local_mask]
                local_rust = rust[y_start:y_end, x_start:x_end][local_mask] > 0
                local_dark = dark[y_start:y_end, x_start:x_end][local_mask] > 0
                features.extend(local_lab.mean(axis=0).tolist())
                features.extend(
                    [
                        float(local_hsv[:, 0].mean() / 179.0),
                        float(local_gradient.mean()),
                        float(local_rust.mean()),
                        float(local_dark.mean()),
                    ]
                )
            else:
                features.extend([0.0] * 7)

    # Contour geometry and normalized silhouette-width profile.
    contours, _ = cv2.findContours(
        shell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea) if contours else None
    if contour is not None:
        bx, by, box_width, box_height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        hu = cv2.HuMoments(cv2.moments(contour)).flatten()
        hu = (np.sign(hu) * np.log10(np.abs(hu) + 1e-30)).tolist()
    else:
        box_width, box_height = 1, 1
        area = perimeter = hull_area = 0.0
        hu = [0.0] * 7

    widths = np.count_nonzero(shell_mask, axis=1).astype(np.float32)
    max_width = max(1.0, float(widths.max()))
    profile = []
    for row in range(8):
        band = widths[row * 24 : (row + 1) * 24]
        profile.append(float(band.mean() / max_width))
    features.extend(
        [
            area / max(1, box_width * box_height),
            box_width / max(1, box_height),
            area / max(1.0, hull_area),
            4 * math.pi * area / max(1.0, perimeter * perimeter),
            *profile,
            *hu,
        ]
    )

    # Rust connected components provide explicit area, span, and streak cues.
    rust_closed = cv2.morphologyEx(
        rust,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        rust_closed, connectivity=8
    )
    components = stats[1:component_count]
    shell_area = max(1, cv2.countNonZero(shell_mask))
    occupied_rows = np.flatnonzero(widths > 0)
    shell_height = (
        max(1, int(occupied_rows[-1] - occupied_rows[0] + 1))
        if len(occupied_rows)
        else 1
    )
    max_area = int(components[:, cv2.CC_STAT_AREA].max()) if len(components) else 0
    max_height = int(components[:, cv2.CC_STAT_HEIGHT].max()) if len(components) else 0
    streaks = []
    for component in components:
        component_width = int(component[cv2.CC_STAT_WIDTH])
        component_height = int(component[cv2.CC_STAT_HEIGHT])
        component_area = int(component[cv2.CC_STAT_AREA])
        elongation = component_height / max(1, component_width)
        streaks.append(
            (component_height / shell_height)
            * min(1.0, elongation / 3.0)
            * math.sqrt(component_area / shell_area)
        )
    features.extend(
        [
            cv2.countNonZero(rust) / shell_area,
            cv2.countNonZero(dark) / shell_area,
            max_area / shell_area,
            max_height / shell_height,
            max(streaks, default=0.0),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def robust_standardize(features: np.ndarray) -> np.ndarray:
    """Scale each feature using median/MAD so outliers do not set the scale."""
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    scale = np.maximum.reduce((1.4826 * mad, iqr_scale, np.full_like(mad, 0.02)))
    return np.clip((features - median) / scale, -10.0, 10.0)


def fit_gaussian(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a regularized Gaussian baseline and return mean and inverse covariance."""
    mean = samples.mean(axis=0)
    covariance = np.cov(samples, rowvar=False)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.75 * covariance + 0.25 * diagonal
    ridge = max(1e-5, float(np.trace(covariance) / len(mean)) * 1e-4)
    covariance += np.eye(len(mean), dtype=np.float64) * ridge
    inverse = np.linalg.pinv(covariance, hermitian=True)
    return mean, inverse


def mahalanobis_squared(
    samples: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray
) -> np.ndarray:
    delta = samples - mean
    return np.einsum("ij,jk,ik->i", delta, inverse_covariance, delta)


def monte_carlo_scores(
    features: np.ndarray, simulations: int = 128, seed: int = 2026
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Bootstrap a robust good-shell baseline and estimate empirical tail scores."""
    standardized = robust_standardize(features).astype(np.float64)

    # Start with the central 95%, then repeatedly trim the largest distances.
    centrality = np.einsum("ij,ij->i", standardized, standardized)
    good = centrality <= np.quantile(centrality, 0.95)
    for _ in range(4):
        mean, inverse = fit_gaussian(standardized[good])
        distance = mahalanobis_squared(standardized, mean, inverse)
        updated = distance <= np.quantile(distance, 0.95)
        if np.array_equal(updated, good):
            break
        good = updated

    good_indices = np.flatnonzero(good)
    if len(good_indices) <= standardized.shape[1]:
        raise ValueError("Too few baseline samples for the feature dimension")

    rng = np.random.default_rng(seed)
    simulated_distances = np.empty((len(features), simulations), dtype=np.float32)
    simulated_p_values = np.empty((len(features), simulations), dtype=np.float32)
    thresholds = []
    for simulation in range(simulations):
        bootstrap = rng.choice(good_indices, size=len(good_indices), replace=True)
        in_bootstrap = np.zeros(len(features), dtype=bool)
        in_bootstrap[np.unique(bootstrap)] = True
        out_of_bag = good_indices[~in_bootstrap[good_indices]]

        mean, inverse = fit_gaussian(standardized[bootstrap])
        distance = mahalanobis_squared(standardized, mean, inverse)
        simulated_distances[:, simulation] = distance
        if len(out_of_bag):
            null = np.sort(distance[out_of_bag])
            tail_count = len(null) - np.searchsorted(null, distance, side="left")
            simulated_p_values[:, simulation] = (tail_count + 1) / (len(null) + 1)
            thresholds.append(float(np.quantile(null, 0.99)))
        else:
            simulated_p_values[:, simulation] = 1.0

    # Median over simulations limits dependence between bootstrap replicates.
    observed = np.median(simulated_distances, axis=1)
    p_values = np.median(simulated_p_values, axis=1)
    threshold = float(np.median(thresholds))
    return p_values, observed, threshold, len(good_indices)


def monte_carlo_surface_scores(
    paths: list[Path], samples: int = 10_000, seed: int = 2026
) -> np.ndarray:
    """Estimate anomalous surface-pixel rates against the dataset baseline.

    Each simulation samples the same random shell-surface coordinates from every
    image. A sample is a hit when its green-minus-red color, high-pass green
    texture, or local shell width is unusually far from the robust population
    baseline. The per-shell hit fraction is the Monte Carlo anomaly score.
    """
    rng = np.random.default_rng(seed)
    row_samples = rng.integers(0, 128, size=samples)
    col_samples = rng.integers(0, 32, size=samples)
    green_red: list[np.ndarray] = []
    highpass_green: list[np.ndarray] = []
    widths: list[np.ndarray] = []

    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        height, image_width = bgr.shape[:2]
        b, g, r = [channel.astype(np.int16) for channel in cv2.split(bgr)]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, saturation, value = cv2.split(hsv)
        mask = (
            (h >= 38)
            & (h <= 68)
            & (saturation >= 38)
            & (value >= 20)
            & (value <= 245)
            & ((g - np.maximum(r, b)) >= 8)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        shell = np.zeros_like(mask)
        for component in range(1, count):
            area = stats[component, cv2.CC_STAT_AREA]
            component_width = stats[component, cv2.CC_STAT_WIDTH]
            center_x = centroids[component, 0]
            if (
                0.37 * image_width <= center_x <= 0.63 * image_width
                and area >= 1000
                and component_width <= 160
            ):
                shell[labels == component] = 255
        if cv2.countNonZero(shell) == 0:
            shell = segment_shell(bgr)

        ys, xs = np.where(shell > 0)
        if len(xs) == 0:
            raise ValueError(f"Could not isolate shell in {path}")
        center_x, center_y = float(xs.mean()), float(ys.mean())
        _, eigenvectors = np.linalg.eigh(np.cov(xs - center_x, ys - center_y))
        vx, vy = eigenvectors[:, 1]
        if vy < 0:
            vx, vy = -vx, -vy
        angle = -math.degrees(math.atan2(float(vx), float(vy)))
        matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotated = cv2.warpAffine(
            bgr, matrix, (image_width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        shell = cv2.warpAffine(
            shell, matrix, (image_width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ) > 0

        row_counts = shell.sum(axis=1)
        valid = np.flatnonzero(row_counts >= 10)
        if len(valid) < 2:
            raise ValueError(f"Shell alignment failed for {path}")
        y0, y1 = int(valid[0]) + 2, int(valid[-1]) - 4
        left_edge = np.argmax(shell, axis=1)
        right_edge = image_width - 1 - np.argmax(shell[:, ::-1], axis=1)
        valid_edges = np.flatnonzero(
            (row_counts >= 10)
            & ((np.arange(height) < y0 + 180) | ((right_edge - left_edge) >= 80))
        )
        rows = np.arange(y0, y1 + 1)
        if len(rows) < 2 or len(valid_edges) < 2:
            raise ValueError(f"Shell surface extraction failed for {path}")
        left = np.interp(rows, valid_edges, left_edge[valid_edges])
        right = np.interp(rows, valid_edges, right_edge[valid_edges])
        row_positions = np.linspace(0, len(rows) - 1, 128)
        left = np.interp(row_positions, np.arange(len(rows)), left)
        right = np.interp(row_positions, np.arange(len(rows)), right)
        across = np.linspace(0.05, 0.95, 32)[None, :]
        map_x = (left[:, None] + across * (right - left)[:, None]).astype(np.float32)
        map_y = np.repeat((y0 + row_positions)[:, None], 32, axis=1).astype(np.float32)
        surface = cv2.remap(rotated, map_x, map_y, cv2.INTER_LINEAR).astype(np.float32)

        green_channel = surface[:, :, 1]
        green_minus_red = green_channel - surface[:, :, 2]
        highpass = green_channel - cv2.GaussianBlur(
            green_channel, (1, 15), 3.0
        )
        width_profile = (right - left) / max(1, len(rows)) * 500.0
        green_red.append(green_minus_red[row_samples, col_samples])
        highpass_green.append(highpass[row_samples, col_samples])
        widths.append(width_profile[row_samples])

    color = np.stack(green_red)
    texture = np.stack(highpass_green)
    width = np.stack(widths)

    def robust_deviation(values: np.ndarray, floor: float) -> np.ndarray:
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        return np.abs(values - median) / (1.4826 * mad + floor)

    hits = (
        (robust_deviation(color, 1.5) > 4.0)
        | (robust_deviation(texture, 1.5) > 4.0)
        | (robust_deviation(width, 0.5) > 4.5)
    )
    return hits.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()
    if args.simulations < 2:
        parser.error("--simulations must be at least 2")

    paths = unique_images()
    scores = monte_carlo_surface_scores(paths, samples=10_000, seed=args.seed)
    ranking = np.argsort(scores)[::-1]
    candidates = [f"images/{paths[index].name}" for index in ranking[:EXPECTED_COUNT]]

    print(f"Unique images: {len(paths)}")
    print("Monte Carlo surface samples per shell: 10000")
    print("Top anomaly candidates (fraction of anomalous surface samples):")
    for index in ranking[:EXPECTED_COUNT]:
        print(f"images/{paths[index].name}\tscore={scores[index]:.4f}")

    if args.write_output:
        OUTPUT_PATH.write_text("\n".join(candidates) + "\n", encoding="utf-8")
        print(f"Wrote {EXPECTED_COUNT} paths to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
