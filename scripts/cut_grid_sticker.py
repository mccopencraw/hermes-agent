#!/usr/bin/env python3
"""
Cut NxM grid image into individual sticker variants with:
1. Background removed (foreground kept, background made transparent)
2. White border following the sticker outline (dilated alpha mask - original mask)

Usage: python cut_grid_sticker.py <input_image> [output_dir] [--border 20] [--blur 5]
"""

import sys
import os
import argparse
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d

def detect_split_positions(gray, axis='h', num_splits=2):
    """Detect white separator bands in a grid image."""
    h, w = gray.shape[:2]
    
    if axis == 'h':
        margin = int(h * 0.1)
        profile = np.array([np.mean(gray[y, :] > 200) for y in range(h)])
    else:
        margin = int(w * 0.1)
        profile = np.array([np.mean(gray[:, x] > 200) for x in range(w)])
    
    # Smooth profile
    profile = uniform_filter1d(profile, size=11)
    
    # Exclude edge margins
    if axis == 'h':
        profile[:margin] = 0
        profile[-margin:] = 0
    else:
        profile[:margin] = 0
        profile[-margin:] = 0
    
    # Find white bands (ratio > 0.55, mean > 0.75)
    white_mask = profile > 0.75
    positions = []
    in_band = False
    start = 0
    
    for i in range(len(white_mask)):
        if white_mask[i] and not in_band:
            start = i
            in_band = True
        elif (not white_mask[i]) and in_band:
            band_width = i - start
            band_center = start + band_width // 2
            if band_width > (h if axis == 'h' else w) / 15:
                positions.append((start, i, band_center))
            in_band = False
    
    if in_band:
        band_width = len(white_mask) - start
        band_center = start + band_width // 2
        if band_width > (h if axis == 'h' else w) / 15:
            positions.append((start, len(white_mask), band_center))
    
    # Sort by quality and pick top num_splits
    positions.sort(key=lambda p: p[1] - p[0], reverse=True)
    selected = positions[:num_splits]
    selected.sort(key=lambda p: p[2])
    
    return [p[2] for p in selected]

def compute_alpha_mask(img, bg_threshold=250):
    """Compute alpha mask from image - assumes white/light background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # For light backgrounds: foreground is darker
    alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
    
    # Find the background threshold using edge analysis
    h, w = gray.shape
    edge_size = min(h, w) // 20
    
    # Sample edges to determine background brightness
    edges = []
    if edge_size > 0:
        edges.append(gray[:edge_size, :].flatten())
        edges.append(gray[-edge_size:, :].flatten())
        edges.append(gray[:, :edge_size].flatten())
        edges.append(gray[:, -edge_size:].flatten())
    
    bg_sample = np.concatenate(edges)
    bg_median = np.median(bg_sample)
    
    # Create mask: pixels significantly darker than background are foreground
    threshold = bg_median - 30  # Adjust this sensitivity
    threshold = max(threshold, 220)  # Minimum threshold for very light backgrounds
    
    fg_mask = gray < threshold
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    alpha = fg_mask
    
    return alpha, threshold

def create_outline_border(img, alpha, border_width=20, blur_radius=5):
    """Create white border following sticker outline."""
    if np.all(alpha == 0):
        return img, alpha
    
    # Dilate the alpha mask to create border area
    kernel_size = int(border_width * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(alpha, kernel, iterations=1)
    
    # Border = dilated - original
    border_mask = cv2.subtract(dilated, alpha)
    
    # Apply Gaussian blur to border for smooth edges
    if blur_radius > 0:
        border_mask = cv2.GaussianBlur(border_mask, (0, 0), blur_radius)
    
    # Convert to float for alpha blending
    img_float = img.astype(np.float32)
    border_color = np.array([255, 255, 255], dtype=np.float32)
    
    # Where there's border, replace with white
    border_alpha = border_mask.astype(np.float32) / 255.0
    for c in range(3):
        img_float[:, :, c] = img_float[:, :, c] * (1 - border_alpha) + border_color[c] * border_alpha
    
    result = np.clip(img_float, 0, 255).astype(np.uint8)
    
    return result, dilated

def process_grid_image(input_path, output_dir, border_width=20, blur_radius=5, 
                       num_rows=None, num_cols=None):
    """Process grid image into individual stickers."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read {input_path}")
        sys.exit(1)
    
    h, w = img.shape[:2]
    print(f"Processing {input_path} ({w}x{h})")
    print(f"Border width: {border_width}px, Blur radius: {blur_radius}")
    
    # Detect split positions
    h_splits = detect_split_positions(img, 'h', num_rows - 1 if num_rows else 0)
    v_splits = detect_split_positions(img, 'v', num_cols - 1 if num_cols else 0)
    
    # Determine grid dimensions
    if num_rows is None:
        num_rows = len(h_splits) + 1
    if num_cols is None:
        num_cols = len(v_splits) + 1
    
    print(f"Grid: {num_rows}x{num_cols}")
    print(f"Horizontal splits: {h_splits}")
    print(f"Vertical splits: {v_splits}")
    
    # Create split points
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
            
            # Create alpha mask (remove background)
            alpha, threshold = compute_alpha_mask(cell)
            
            # Create outline border
            bordered, new_alpha = create_outline_border(cell, alpha, border_width, blur_radius)
            
            # Combine into BGRA
            bgra = cv2.merge([bordered, new_alpha])
            
            # Save
            filename = f"sticker_r{row+1}c{col+1}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, bgra)
            print(f"  Saved {filename}")
            sticker_files.append(output_path)
    
    # Create preview grid
    create_preview_grid(sticker_files, num_rows, num_cols, os.path.join(output_dir, "preview_grid.png"))
    
    print(f"\nDone! {len(sticker_files)} stickers created in {output_dir}")
    return sticker_files

def create_preview_grid(sticker_files, rows, cols, output_path, bg_color=(240, 240, 240)):
    """Create preview image showing all stickers on light background."""
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
            # Resize maintaining aspect ratio
            h, w = sticker.shape[:2]
            scale = min(cell_size / h, cell_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            y = padding + row * (cell_size + padding) + (cell_size - new_h) // 2
            x = padding + col * (cell_size + padding) + (cell_size - new_w) // 2
            
            # Alpha blend
            alpha = resized[:, :, 3:].astype(np.float32) / 255.0
            for c in range(3):
                preview[y:y+new_h, x:x+new_w, c] = (
                    preview[y:y+new_h, x:x+new_w, c] * (1 - alpha[:, :, 0]) +
                    resized[:, :, c] * alpha[:, :, 0]
                )
    
    cv2.imwrite(output_path, preview)
    print(f"Preview saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut grid image into sticker variants with outline borders")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", nargs="?", default="./stickers", help="Output directory")
    parser.add_argument("--border", type=int, default=20, help="Border width in pixels (default: 20)")
    parser.add_argument("--blur", type=int, default=5, help="Border blur radius (default: 5)")
    parser.add_argument("--rows", type=int, help="Force number of rows")
    parser.add_argument("--cols", type=int, help="Force number of columns")
    
    args = parser.parse_args()
    
    process_grid_image(
        args.input, args.output,
        border_width=args.border,
        blur_radius=args.blur,
        num_rows=args.rows,
        num_cols=args.cols
    )