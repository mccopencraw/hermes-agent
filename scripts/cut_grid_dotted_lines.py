#!/usr/bin/env python3
"""
Cut grid image along black dotted/dash lines.
Removes white background up to the dotted line boundaries, extracting individual characters.

Usage: python cut_grid_dotted_lines.py <input_image> [output_dir] [--rows 3] [--cols 3]
"""

import sys
import os
import argparse
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d, gaussian_filter

def detect_dotted_line_positions(gray, axis='h', num_splits=2):
    """Detect black dotted/dash lines in a grid image."""
    h, w = gray.shape[:2]

    if axis == 'h':
        margin = int(h * 0.1)
        # Sample a band across the middle (or multiple bands for safety)
        band_size = min(h // 8, 50)
        bands = []
        for y_offset in [-h//6, 0, h//6]:
            y = h // 2 + y_offset
            band = gray[max(0, y-band_size):min(h, y+band_size), :].mean(axis=0)
            bands.append(band)
        profile = np.mean(bands, axis=0)
        # Invert: dark lines become high values
        profile = 255 - profile.astype(float)
    else:
        margin = int(w * 0.1)
        band_size = min(w // 8, 50)
        bands = []
        for x_offset in [-w//6, 0, w//6]:
            x = w // 2 + x_offset
            band = gray[:, max(0, x-band_size):min(w, x+band_size)].mean(axis=1)
            bands.append(band)
        profile = np.mean(bands, axis=0)
        # Invert
        profile = 255 - profile.astype(float)

    # Smooth profile
    profile = gaussian_filter(profile, sigma=3)

    # Exclude edge margins
    profile[:margin] = 0
    profile[-margin:] = 0

    # Normalize
    if profile.max() > 0:
        profile = profile / profile.max()

    # Find peaks (dark lines = high values in inverted profile)
    # Use threshold - look for significant dark lines
    threshold = 0.4
    dark_mask = profile > threshold

    positions = []
    in_line = False
    start = 0

    for i in range(len(dark_mask)):
        if dark_mask[i] and not in_line:
            start = i
            in_line = True
        elif (not dark_mask[i]) and in_line:
            line_width = i - start
            line_center = start + line_width // 2
            if line_width > 5:  # Minimum line width
                positions.append((start, i, line_center))
            in_line = False

    if in_line:
        line_width = len(dark_mask) - start
        if line_width > 5:
            positions.append((start, len(dark_mask), start + line_width // 2))

    # Sort by strength (average profile value in the band)
    positions.sort(key=lambda p: np.mean(profile[p[0]:p[1]]), reverse=True)

    # Pick top num_splits, ensuring they're well-separated
    dim = h if axis == 'h' else w
    min_separation = dim // (num_splits + 2)

    selected = []
    for pos in positions:
        # Check distance from already selected
        if all(abs(pos[2] - s[2]) > min_separation for s in selected):
            selected.append(pos)
            if len(selected) >= num_splits:
                break

    selected.sort(key=lambda p: p[2])
    return [p[2] for p in selected]


def create_character_mask(img, border_size=5):
    """Create mask for character region, removing white background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Determine background color from edges
    h, w = gray.shape
    edge_size = min(h, w) // 10
    edges = np.concatenate([
        gray[:edge_size, :].flatten(),
        gray[-edge_size:, :].flatten(),
        gray[:, :edge_size].flatten(),
        gray[:, -edge_size:].flatten()
    ])

    bg_median = np.median(edges)

    # Threshold to find non-background (characters + lines)
    threshold = bg_median - 30
    threshold = max(threshold, 200)  # For white background

    mask = (gray < threshold).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove line artifacts (thin dark lines that aren't part of character)
    # Use area-based filtering
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_char_area = (h * w) * 0.02  # Character should be at least 2% of the area
    char_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_char_area:
            cv2.drawContours(char_mask, [contour], -1, 255, -1)

    return char_mask


def cut_character_with_dotted_boundary(img, dotted_mask, char_mask):
    """Cut character following the dotted line boundary, removing background."""
    h, w = img.shape[:2]

    # Create final mask: character area, respecting dotted boundary
    # Use dotted_mask as a cutting guide - areas outside the character but inside dotted boundaries
    # should be removed

    # If we have a dotted line on the left/right/top/bottom of this cell,
    # use it as a hard boundary
    # For simplicity, the char_mask already excludes white background
    # The dotted lines serve as guide for where the character ends

    return char_mask


def process_grid_dotted(input_path, output_dir, num_rows=3, num_cols=3, border_width=20):
    """Process grid image with dotted line cutting."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read {input_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Processing {input_path} ({w}x{h})")
    print(f"Grid: {num_rows}x{num_cols}")

    # Detect split positions
    h_splits = detect_dotted_line_positions(img, 'h', num_rows - 1)
    v_splits = detect_dotted_line_positions(img, 'v', num_cols - 1)

    print(f"Horizontal dotted lines detected: {len(h_splits)} at {h_splits}")
    print(f"Vertical dotted lines detected: {len(v_splits)} at {v_splits}")

    # Create split points (include borders)
    h_points = [0] + sorted(h_splits) + [h]
    v_points = [0] + sorted(v_splits) + [w]

    os.makedirs(output_dir, exist_ok=True)

    sticker_files = []

    for row in range(num_rows):
        for col in range(num_cols):
            y1, y2 = h_points[row], h_points[row + 1]
            x1, x2 = v_points[col], v_points[col + 1]

            # Extract cell
            cell = img[y1:y2, x1:x2]

            # Create character mask (removes white background)
            char_mask = create_character_mask(cell)

            # Create output image with transparent background
            bgra = cv2.merge([cell, char_mask])

            # Save
            filename = f"char_r{row+1}c{col+1}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, bgra)
            print(f"  Saved {filename}")
            sticker_files.append(output_path)

    # Create preview
    create_preview(sticker_files, num_rows, num_cols, os.path.join(output_dir, "preview_chars.png"))

    print(f"\nDone! {len(sticker_files)} characters created in {output_dir}")
    return sticker_files


def create_preview(sticker_files, rows, cols, output_path, bg_color=(240, 240, 240)):
    """Create preview grid."""
    cell_size = 300
    padding = 20

    preview = np.ones((rows * (cell_size + padding) + padding,
                      cols * (cell_size + padding) + padding, 3),
                     dtype=np.uint8) * bg_color

    for i, sticker_path in enumerate(sticker_files):
        row = i // cols
        col = i % cols

        sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
        if sticker is None:
            continue

        if sticker.shape[2] == 4:
            h, w = sticker.shape[:2]
            scale = min(cell_size / h, cell_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_AREA)

            y = padding + row * (cell_size + padding) + (cell_size - new_h) // 2
            x = padding + col * (cell_size + padding) + (cell_size - new_w) // 2

            alpha = resized[:, :, 3:].astype(np.float32) / 255.0
            for c in range(3):
                preview[y:y+new_h, x:x+new_w, c] = (
                    preview[y:y+new_h, x:x+new_w, c] * (1 - alpha[:, :, 0]) +
                    resized[:, :, c] * alpha[:, :, 0]
                )

    cv2.imwrite(output_path, preview)
    print(f"Preview saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut grid image along dotted lines")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", nargs="?", default="./characters", help="Output directory")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns (default: 3)")
    parser.add_argument("--border", type=int, default=20, help="White border width (default: 20)")

    args = parser.parse_args()

    process_grid_dotted(
        args.input, args.output,
        num_rows=args.rows,
        num_cols=args.cols
    )