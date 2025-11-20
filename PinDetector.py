import cv2
import numpy as np
import os
import math
def create_bw_image(INPUT_PATH):

    # tunables (safe defaults)
    CLAHE_CLIP  = 2.0
    CLAHE_TILE  = (8, 8)

    NLM_H       = 14          # 10–25; higher removes more grain
    BILAT_D     = 5           # 5–9 neighborhood
    BILAT_SC    = 135          # sigmaColor (edge-preserving)
    BILAT_SS    = 20          # sigmaSpace

    UNSHARP_A   = 1.4         # addWeighted: amount on base
    UNSHARP_B   = -0.4        # addWeighted: amount on blur
    GAUSS_FOR_US = (3, 3)     # blur kernel for unsharp

    ADAPT_BLOCK = 31          # odd, ~chip text size or larger
    ADAPT_C     = 5           # subtractive constant

    # --- main ---
    img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(INPUT_PATH)

    # 1) CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    im1 = clahe.apply(img)
    #save("01_clahe.png", im1)

    # 2) Edge-preserving denoise
    im2 = cv2.fastNlMeansDenoising(im1, None, h=NLM_H, templateWindowSize=7, searchWindowSize=21)
    im2 = cv2.bilateralFilter(im2, d=BILAT_D, sigmaColor=BILAT_SC, sigmaSpace=BILAT_SS)
    #save("02_denoised.png", im2)

    # 3) Mild unsharp mask
    blur = cv2.GaussianBlur(im2, GAUSS_FOR_US, 0)
    im3 = cv2.addWeighted(im2, UNSHARP_A, blur, UNSHARP_B, 0)
    #save("03_unsharp.png", im3)

    # 4) Adaptive threshold (chip body becomes black)
    adaptive_bw = cv2.adaptiveThreshold(im3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, ADAPT_BLOCK, ADAPT_C)

    return adaptive_bw


def make_bw(img, thresh_value=120):
    # Load grayscale
    if img is None:
        raise RuntimeError("Could not read input image")

    # Apply threshold
    _, bw = cv2.threshold(
        img,
        thresh_value,   # user-controlled
        255,
        cv2.THRESH_BINARY
    )

    return bw

import cv2
import numpy as np

def fill_largest_white(img):
    """
    Load a black/white image and fill the largest white area with black.
    """
    # 1) Load as grayscale
    if img is None:
        raise RuntimeError("Could not read input image")

    # 2) Ensure binary (0 or 255) just in case
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3) Create a mask of white pixels (255 -> 1, 0 -> 0)
    white_mask = (bw == 255).astype(np.uint8)

    # 4) Connected components to find white regions
    #    connectivity=8: diagonals are connected too
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    # label 0 is background; skip it
    if num_labels <= 1:
        return bw

    # 5) Find the largest white component (by area)
    # stats[:, cv2.CC_STAT_AREA] gives the area
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_label = 1 + np.argmax(areas)  # shift back by 1

    # 6) Fill that component black in the original image
    result = bw.copy()
    result[labels == largest_label] = 0  # set to black
    final = 255 - result
    # 7) Save
    return final



def merge_black_regions(src_bw, adaptive_bw):

    if src_bw is None or adaptive_bw is None:
        raise RuntimeError("Could not load input images")

    if src_bw.shape != adaptive_bw.shape:
        raise RuntimeError("Images must have the same resolution")

    # Black = 0, White = 255
    # If src_bw is black → force output black
    merged = adaptive_bw.copy()
    merged[src_bw == 0] = 0

    # Save
    return merged

import cv2
import numpy as np

def find_chip_with_pins_bbox(img):
    """
    Very sensitive chip+pins bounding box detection.
    Finds the full rectangular region covering body + all pins.
    Returns (x, y, w, h).
    """
    if img is None:
        raise RuntimeError(f"Could not read input image.")


    # ----------------------------------------------------------------------
    # 1. VERY SENSITIVE THRESHOLDING
    #    Adaptive is more sensitive than Otsu for tiny pin edges.
    # ----------------------------------------------------------------------
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,   # small block → reacts to small changes
        5     # C value → keeps thin edges
    )

    # ----------------------------------------------------------------------
    # 2. MORPHOLOGY (boost sensitivity to pins)
    # ----------------------------------------------------------------------

    # Thicken thin pins (sensitive dilation)
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, dil_k, iterations=2)

    # Close tiny gaps between pin segments
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)

    # Optional: remove tiny noise specks
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)

    # ----------------------------------------------------------------------
    # 3. FIND FULL CHIP CONTOUR
    # ----------------------------------------------------------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found for chip + pins")

    # Full chip = largest contour
    cnt = max(contours, key=cv2.contourArea)

    # Use convex hull → avoids small pin gaps giving a broken contour
    hull = cv2.convexHull(cnt)

    # ----------------------------------------------------------------------
    # 4. GET BOUNDING BOX
    # ----------------------------------------------------------------------
    x, y, w, h = cv2.boundingRect(hull)
    out = img.copy()

    return out, x, y, w, h

def fill_outside_bbox(img, bbox_x, bbox_y, bbox_w, bbox_h):
    """
    Removes small black noise connected components in a binary image,
    BUT ONLY OUTSIDE the given bounding box (e.g. chip + pins).

    bbox_x, bbox_y, bbox_w, bbox_h : bounding box around chip+pins.
    max_area = max pixel area of a black component considered 'noise'.
    """
 # Load grayscale binary
    if img is None:
        raise RuntimeError("Could not read input image")

    # Ensure binary
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Create an all-white image
    cleaned = np.full_like(bw, 255)

    # Restore only the requested bounding box region
    x0, y0, w, h = bbox_x, bbox_y, bbox_w, bbox_h
    cleaned[y0:y0+h, x0:x0+w] = bw[y0:y0+h, x0:x0+w]

    return cleaned

import cv2
import numpy as np

def fill_white_inside_black(img, kx=5, ky=5, iterations=1):

    if img is None:
        raise RuntimeError("Could not read input image")

    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 2) Invert: chip+pins = white, background = black
    inv = 255 - bw

    # 3) Morphological closing to fill gaps/holes in chip+pins
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    inv_closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel,
                                  iterations=iterations)

    # 4) Everything that is white in inv_closed is "chip region"
    chip_mask = inv_closed == 255

    # 5) In the ORIGINAL image, force that region to black
    result = bw.copy()
    result[chip_mask] = 0

    return result

import cv2
import numpy as np

def fill_small_white_areas(img, max_white_area=4000):
    
    if img is None:
        raise RuntimeError("Could not read input image")

    # Ensure binary
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 2) Create mask where white = 1
    white_mask = (bw == 255).astype(np.uint8)

    # 3) Connected components on white regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    # 4) Copy image and fill small white components to black
    result = bw.copy()

    for label in range(1, num_labels):  # skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area < max_white_area:
            # set this white component to black
            result[labels == label] = 0

    return result

import cv2
import numpy as np
import math

def rotate_chip_to_horizontal(img, img7_fill_small_white):
    # 1) Read image
    if img is None:
        raise RuntimeError("Could not read input image")

    gray = img

    # 2) Threshold to get the chip body
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) Find largest contour
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chip_cnt = max(contours, key=cv2.contourArea)

    # Min area rect
    (cx, cy), (w, h), angle = cv2.minAreaRect(chip_cnt)

    # Ensure angle refers to long side
    if w < h:
        angle += 90.0

    #print("Original angle:", angle)

    while angle >= 90.0:
        if (angle - 90.0) < 0:
            break
        angle -= 90.0

    # Safety clamp
    if angle < 0:
        angle = 0

    #print("Normalized angle:", angle)
    # -------------------------------------------------------

    # 4) Rotate by normalized angle
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    img = img7_fill_small_white
    rotated = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated, angle


def calculate_frequency_switches(region):
    """Calculate the number of black/white transitions in a region"""
    if region.size == 0:
        return float('inf')
    
    # Count horizontal transitions
    h_diff = np.abs(np.diff(region.astype(int), axis=1))
    h_switches = np.sum(h_diff > 127)
    
    # Count vertical transitions
    v_diff = np.abs(np.diff(region.astype(int), axis=0))
    v_switches = np.sum(v_diff > 127)
    
    # Normalize by area
    total_switches = h_switches + v_switches
    area = region.shape[0] * region.shape[1]
    return total_switches / area if area > 0 else float('inf')

def calculate_border_black_ratio(region):
    if region.size == 0:
        return 0
    
    h, w = region.shape
    if h < 3 or w < 3:
        return 0
    
    # Extract border pixels (top, bottom, left, right edges)
    top_border = region[0, :]
    bottom_border = region[-1, :]
    left_border = region[:, 0]
    right_border = region[:, -1]
    
    # Combine all border pixels
    border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    
    # Count black pixels (value < 127 after threshold inversion)
    black_count = np.sum(border_pixels < 127)
    total_border_pixels = len(border_pixels)
    
    return black_count / total_border_pixels if total_border_pixels > 0 else 0

def compute_bbox_body(img):
    # Read the image
    gray = img

    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Create a version with heavy erosion to remove pins and keep only the chip body
    kernel_large = np.ones((15, 15), np.uint8)
    eroded = cv2.erode(binary, kernel_large, iterations=2)

    # Find contours in the eroded image (should be just the chip body)
    contours_eroded, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the best chip body candidate based on frequency and border criteria
    chip_x, chip_y, chip_w, chip_h = 0, 0, 0, 0
    best_score = -1
    best_border_ratio = 0
    best_freq = 0

    for contour in contours_eroded:
        area = cv2.contourArea(contour)
        
        # Only consider large contours
        if area < 10000:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the region from original grayscale
        region = gray[y:y+h, x:x+w]
        
        # Calculate frequency switches (lower is better - want smooth interior)
        switch_freq = calculate_frequency_switches(region)
        
        # Calculate border black ratio (higher is better - want black border)
        border_black_ratio = calculate_border_black_ratio(region)
        
        # Combined score: high border black ratio and low frequency switches
        # Normalize and combine (border ratio is 0-1, freq needs to be inverted)
        freq_score = 1.0 / (1.0 + switch_freq * 100)  # Lower freq = higher score
        combined_score = border_black_ratio * 0.6 + freq_score * 0.4
        
        if combined_score > best_score:
            best_score = combined_score
            best_border_ratio = border_black_ratio
            best_freq = switch_freq
            chip_x, chip_y, chip_w, chip_h = x, y, w, h

    if best_score == -1:
        print("Could not find main chip body!")
        exit()

    # Expand the bounding box slightly to get the actual chip body
    expansion = 10
    chip_x = max(0, chip_x - expansion)
    chip_y = max(0, chip_y - expansion)
    chip_w = min(img.shape[1] - chip_x, chip_w + 2 * expansion)
    chip_h = min(img.shape[0] - chip_y, chip_h + 2 * expansion)

    # Refine the border to achieve at least 70% black ratio AND low frequency
    target_black_ratio = 0.85
    target_frequency = 0.02  # Maximum acceptable frequency
    max_iterations = 10000
    step_size = 10

    for iteration in range(max_iterations):
        region = gray[chip_y:chip_y+chip_h, chip_x:chip_x+chip_w]
        current_black_ratio = calculate_border_black_ratio(region)
        current_frequency = calculate_frequency_switches(region)
        
        # Stop if we meet both criteria
        if current_black_ratio >= target_black_ratio and current_frequency <= target_frequency:
            print(f"Target criteria achieved after {iteration} iterations")
            break
        
        # Check minimum size
        h, w = region.shape
        if h < 20 or w < 20:
            print(f"Minimum size reached at iteration {iteration}")
            break
        
        # Calculate white pixel counts on each border
        top_white = np.sum(region[0, :] > 127)
        bottom_white = np.sum(region[-1, :] > 127)
        left_white = np.sum(region[:, 0] > 127)
        right_white = np.sum(region[:, -1] > 127)
        
        # Determine which sides to shrink
        # Priority: shrink sides with most white pixels, especially if black ratio is low
        shrink_threshold = 0.2  # Shrink if more than 20% of edge is white
        shrunk = False
        
        if current_black_ratio < target_black_ratio:
            # Aggressively shrink any side with white pixels
            if top_white > w * shrink_threshold and chip_h > 20:
                chip_y += step_size
                chip_h -= step_size
                shrunk = True
            if bottom_white > w * shrink_threshold and chip_h > 20:
                chip_h -= step_size
                shrunk = True
            if left_white > h * shrink_threshold and chip_w > 20:
                chip_x += step_size
                chip_w -= step_size
                shrunk = True
            if right_white > h * shrink_threshold and chip_w > 20:
                chip_w -= step_size
                shrunk = True
        
        # If black ratio is good but frequency is too high, shrink more carefully
        elif current_frequency > target_frequency:
            # Shrink from all sides slightly to get deeper into solid area
            if chip_h > 20 and chip_w > 20:
                chip_y += step_size
                chip_h -= 2 * step_size
                chip_x += step_size
                chip_w -= 2 * step_size
                shrunk = True
        
        if not shrunk:
            print(f"No more adjustments possible at iteration {iteration}")
            break

    # Final metrics
    final_region = gray[chip_y:chip_y+chip_h, chip_x:chip_x+chip_w]
    final_black_ratio = calculate_border_black_ratio(final_region)
    final_frequency = calculate_frequency_switches(final_region)


    # Now find pins in the original binary image
    kernel = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find all contours
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return chip_x, chip_y, chip_w, chip_h

def detect_pins_by_scanning(binary, chip_x, chip_y, chip_w, chip_h,
                            scan_offset=5,
                            scan_shift_to_edge=0,
                            min_pin_thickness=3,
                            margin=2,
                            min_box_w=1,
                            min_box_h=1,
                            min_box_area=0,
                            disconnected_tolerance=15):
    """
    Detect pins by orthogonal scanning from the chip body box.
    """

    H, W = binary.shape
    PIN = 0  # black pixel

    SCAN_OFFSET_REVISED = 50  # revised scan offset for oversized boxes

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # ---------- generic scanning over 1D edge ----------
    def scan_edge(get_pixel_line, coord_range):
        hits = []  # (coord, d_start, d_end)
        for coord in range(*coord_range):
            d = 0
            inside = False
            black_start = None
            while True:
                p = get_pixel_line(coord, d)
                if p is None:
                    if inside:
                        if d - black_start >= min_pin_thickness:
                            hits.append((coord, black_start, d))
                    break

                if p == PIN:
                    if not inside:
                        inside = True
                        black_start = d
                else:
                    if inside:
                        inside = False
                        if d - black_start >= min_pin_thickness:
                            hits.append((coord, black_start, d))
                d += 1
        return hits

    def group_hits(hits, max_gap=2):
        if not hits:
            return []
        hits.sort(key=lambda t: t[0])
        groups = []
        cur_coords = [hits[0][0]]
        cur_starts = [hits[0][1]]
        cur_ends = [hits[0][2]]

        for coord, sd, ed in hits[1:]:
            if coord - cur_coords[-1] <= max_gap:
                cur_coords.append(coord)
                cur_starts.append(sd)
                cur_ends.append(ed)
            else:
                groups.append((cur_coords, cur_starts, cur_ends))
                cur_coords = [coord]
                cur_starts = [sd]
                cur_ends = [ed]

        groups.append((cur_coords, cur_starts, cur_ends))
        return groups

    def check_disconnected_end(get_pixel_line, coords, current_end, tolerance):
        """Check if there's a disconnected pin end within tolerance distance."""
        max_extension = current_end
        for coord in coords:
            for d in range(current_end + 1, current_end + tolerance + 1):
                p = get_pixel_line(coord, d)
                if p is None:
                    break
                if p == PIN:
                    d_ext = d
                    while True:
                        p_next = get_pixel_line(coord, d_ext + 1)
                        if p_next is None or p_next != PIN:
                            break
                        d_ext += 1
                    max_extension = max(max_extension, d_ext + 1)
                    break
        return max_extension

    # per-side pin collections
    top_pins = []
    bottom_pins = []
    left_pins = []
    right_pins = []

    # =====================================================================
    # TOP EDGE (scan upward)
    # =====================================================================
    def get_pixel_top(x, d):
        y_start = chip_y - scan_offset - scan_shift_to_edge
        yy = y_start - d
        if yy < 0:
            return None
        return binary[yy, x]

    top_hits = scan_edge(get_pixel_top, (chip_x, chip_x + chip_w))
    top_groups = group_hits(top_hits)

    for coords, starts, ends in top_groups:
        x1 = min(coords) - 3  # extend width
        x2 = max(coords) + 3  # extend width

        d_start_eff = max(0, min(starts) - scan_shift_to_edge)
        d_end_eff   = max(d_start_eff, max(ends) - scan_shift_to_edge)

        # Check for disconnected pin end
        d_end_extended = check_disconnected_end(get_pixel_top, coords, d_end_eff, disconnected_tolerance)
        
        y_inner = chip_y
        y_outer = chip_y - d_end_extended - margin
        y_outer -= scan_offset
        y_outer = clamp(y_outer, 0, chip_y)

        if y_inner > y_outer:
            x1 = clamp(x1, 0, W - 1)  # clamp after extension
            x2 = clamp(x2, 0, W - 1)
            w = x2 - x1 + 1
            h = y_inner - y_outer + 1
            area = w * h

        # oversize check: width too large
        if w > 120:
            revised_pins = []

            def get_pixel_top_rev(x, d):
                y_start = chip_y - SCAN_OFFSET_REVISED - scan_shift_to_edge
                yy = y_start - d
                if yy < 0:
                    return None
                return binary[yy, x]

            sub_hits = scan_edge(get_pixel_top_rev, (x1, x2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) - scan_shift_to_edge)
                sd_end_eff   = max(sd_start_eff, max(s_ends) - scan_shift_to_edge)

                sd_end_extended = check_disconnected_end(
                    get_pixel_top_rev, s_coords, sd_end_eff, disconnected_tolerance
                )

                y_inner2 = chip_y
                y_outer2 = chip_y - sd_end_extended - margin
                y_outer2 -= SCAN_OFFSET_REVISED
                y_outer2 = clamp(y_outer2, 0, chip_y)

                if y_inner2 <= y_outer2:
                    revised_pins = []
                    break

                sx1 = min(s_coords) - 3
                sx2 = max(s_coords) + 3
                sx1 = clamp(sx1, 0, W - 1)
                sx2 = clamp(sx2, 0, W - 1)
                sw = sx2 - sx1 + 1
                sh = y_inner2 - y_outer2 + 1
                sarea = sw * sh

                if sw < min_box_w or sh < min_box_h or sarea < min_box_area:
                    # fall back to original big box
                    top_pins.append((x1, y_outer, w, h))
                    revised_pins = []
                    break

                revised_pins.append((sx1, y_outer2, sw, sh))

            top_pins.extend(revised_pins)
            continue

        if w < min_box_w or h < min_box_h or area < min_box_area:
            continue

        top_pins.append((x1, y_outer, w, h))

    # =====================================================================
    # BOTTOM EDGE (scan downward)
    # =====================================================================
    def get_pixel_bottom(x, d):
        y_start = chip_y + chip_h + scan_offset + scan_shift_to_edge
        yy = y_start + d
        if yy >= H:
            return None
        return binary[yy, x]

    bottom_hits = scan_edge(get_pixel_bottom, (chip_x, chip_x + chip_w))
    bottom_groups = group_hits(bottom_hits)

    for coords, starts, ends in bottom_groups:
        d_start_eff = max(0, min(starts) - scan_shift_to_edge)
        d_end_eff   = max(d_start_eff, max(ends) - scan_shift_to_edge)

        d_end_extended = check_disconnected_end(
            get_pixel_bottom, coords, d_end_eff, disconnected_tolerance
        )

        y_inner = chip_y + chip_h
        y_outer = chip_y + chip_h + d_end_extended + margin
        y_outer += scan_offset
        y_outer = clamp(y_outer, y_inner, H - 1)

        x1 = min(coords) - 3
        x2 = max(coords) + 3
        x1 = clamp(x1, 0, W - 1)
        x2 = clamp(x2, 0, W - 1)
        w = x2 - x1 + 1
        h = y_outer - y_inner + 1
        area = w * h

        if w > 120:
            revised_pins = []

            def get_pixel_bottom_rev(x, d):
                y_start = chip_y + chip_h + SCAN_OFFSET_REVISED + scan_shift_to_edge
                yy = y_start + d
                if yy >= H:
                    return None
                return binary[yy, x]

            sub_hits = scan_edge(get_pixel_bottom_rev, (x1, x2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) - scan_shift_to_edge)
                sd_end_eff   = max(sd_start_eff, max(s_ends) - scan_shift_to_edge)

                sd_end_extended = check_disconnected_end(
                    get_pixel_bottom_rev, s_coords, sd_end_eff, disconnected_tolerance
                )

                y_inner2 = chip_y + chip_h
                y_outer2 = chip_y + chip_h + sd_end_extended + margin
                y_outer2 += SCAN_OFFSET_REVISED
                y_outer2 = clamp(y_outer2, y_inner2, H - 1)

                sx1 = min(s_coords) - 3
                sx2 = max(s_coords) + 3
                sx1 = clamp(sx1, 0, W - 1)
                sx2 = clamp(sx2, 0, W - 1)
                sw = sx2 - sx1 + 1
                sh = y_outer2 - y_inner2 + 1
                sarea = sw * sh

                if sw < min_box_w or sh < min_box_h or sarea < min_box_area:
                    bottom_pins.append((x1, y_inner, w, h))
                    revised_pins = []
                    break

                revised_pins.append((sx1, y_inner2, sw, sh))

            bottom_pins.extend(revised_pins)
            continue

        if w < min_box_w or h < min_box_h or area < min_box_area:
            continue

        bottom_pins.append((x1, y_inner, w, h))

    # =====================================================================
    # LEFT EDGE (scan left)
    # =====================================================================
    def get_pixel_left(y, d):
        x_start = chip_x - scan_offset - scan_shift_to_edge
        xx = x_start - d
        if xx < 0:
            return None
        return binary[y, xx]

    left_hits = scan_edge(get_pixel_left, (chip_y, chip_y + chip_h))
    left_groups = group_hits(left_hits)

    for coords, starts, ends in left_groups:
        y1 = min(coords) - 3  # extend height
        y2 = max(coords) + 3  # extend height

        d_start_eff = max(0, min(starts) - scan_shift_to_edge)
        d_end_eff   = max(d_start_eff, max(ends) - scan_shift_to_edge)

        # Check for disconnected pin end
        d_end_extended = check_disconnected_end(get_pixel_left, coords, d_end_eff, disconnected_tolerance)

        x_inner = chip_x
        x_outer = chip_x - d_end_extended - margin
        x_outer -= scan_offset
        x_outer = clamp(x_outer, 0, chip_x)

        if x_inner > x_outer:
            y1 = clamp(y1, 0, H - 1)  # clamp after extension
            y2 = clamp(y2, 0, H - 1)
            w = x_inner - x_outer + 1
            h = y2 - y1 + 1
            area = w * h

        if h > 120:
            revised_pins = []

            def get_pixel_left_rev(y, d):
                x_start = chip_x - SCAN_OFFSET_REVISED - scan_offset - scan_shift_to_edge
                xx = x_start - d
                if xx < 0:
                    return None
                return binary[y, xx]

            sub_hits = scan_edge(get_pixel_left_rev, (y1, y2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) - scan_shift_to_edge)
                sd_end_eff   = max(sd_start_eff, max(s_ends) - scan_shift_to_edge)

                sd_end_extended = check_disconnected_end(
                    get_pixel_left_rev, s_coords, sd_end_eff, disconnected_tolerance
                )

                x_inner2 = chip_x
                x_outer2 = chip_x - sd_end_extended - margin
                x_outer2 -= SCAN_OFFSET_REVISED
                x_outer2 = clamp(x_outer2, 0, chip_x)

                if x_inner2 <= x_outer2:
                    revised_pins = []
                    break

                sy1 = min(s_coords) - 3
                sy2 = max(s_coords) + 3
                sy1 = clamp(sy1, 0, H - 1)
                sy2 = clamp(sy2, 0, H - 1)
                sw = x_inner2 - x_outer2 + 1
                sh = sy2 - sy1 + 1
                sarea = sw * sh

                if sw < min_box_h or sh < min_box_w or sarea < min_box_area:
                    left_pins.append((x_outer, y1, w, h))
                    revised_pins = []
                    break

                revised_pins.append((x_outer2, sy1, sw, sh))

            left_pins.extend(revised_pins)
            continue

        if w < min_box_h or h < min_box_w or area < min_box_area:
            continue

        left_pins.append((x_outer, y1, w, h))

    # =====================================================================
    # RIGHT EDGE (scan right)
    # =====================================================================
    def get_pixel_right(y, d):
        x_start = chip_x + chip_w + scan_offset + scan_shift_to_edge
        xx = x_start + d
        if xx >= W:
            return None
        return binary[y, xx]

    right_hits = scan_edge(get_pixel_right, (chip_y, chip_y + chip_h))
    right_groups = group_hits(right_hits)

    for coords, starts, ends in right_groups:
        d_start_eff = max(0, min(starts) - scan_shift_to_edge)
        d_end_eff   = max(d_start_eff, max(ends) - scan_shift_to_edge)

        d_end_extended = check_disconnected_end(
            get_pixel_right, coords, d_end_eff, disconnected_tolerance
        )

        x_inner = chip_x + chip_w
        x_outer = chip_x + chip_w + d_end_extended + margin
        x_outer += scan_offset
        x_outer = clamp(x_outer, x_inner, W - 1)

        y1 = min(coords) - 3
        y2 = max(coords) + 3
        y1 = clamp(y1, 0, H - 1)
        y2 = clamp(y2, 0, H - 1)
        w = x_outer - x_inner + 1
        h = y2 - y1 + 1
        area = w * h

        if h > 120:
            revised_pins = []

            def get_pixel_right_rev(y, d):
                x_start = chip_x + chip_w + SCAN_OFFSET_REVISED + scan_offset + scan_shift_to_edge
                xx = x_start + d
                if xx >= W:
                    return None
                return binary[y, xx]

            sub_hits = scan_edge(get_pixel_right_rev, (y1, y2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) - scan_shift_to_edge)
                sd_end_eff   = max(sd_start_eff, max(s_ends) - scan_shift_to_edge)

                sd_end_extended = check_disconnected_end(
                    get_pixel_right_rev, s_coords, sd_end_eff, disconnected_tolerance
                )

                x_inner2 = chip_x + chip_w
                x_outer2 = chip_x + chip_w + sd_end_extended + margin
                x_outer2 += SCAN_OFFSET_REVISED
                x_outer2 = clamp(x_outer2, x_inner2, W - 1)

                sy1 = min(s_coords) - 3
                sy2 = max(s_coords) + 3
                sy1 = clamp(sy1, 0, H - 1)
                sy2 = clamp(sy2, 0, H - 1)
                sw = x_outer2 - x_inner2 + 1
                sh = sy2 - sy1 + 1
                sarea = sw * sh

                if sw < min_box_h or sh < min_box_w or sarea < min_box_area:
                    right_pins.append((x_inner, y1, w, h))
                    revised_pins = []
                    break

                revised_pins.append((x_inner2, sy1, sw, sh))

            right_pins.extend(revised_pins)
            continue

        if w < min_box_h or h < min_box_w or area < min_box_area:
            continue

        right_pins.append((x_inner, y1, w, h))

    # ---------------------------------------------------------------------
    # Filter: require at least 9 boxes per side
    # ---------------------------------------------------------------------
    pins = []
    pins.append(top_pins if len(top_pins) >= 9 else [])
    pins.append(bottom_pins if len(bottom_pins) >= 9 else [])
    pins.append(left_pins if len(left_pins) >= 9 else [])
    pins.append(right_pins if len(right_pins) >= 9 else [])

    return pins

def detect_defect_pins(pins, img, angle):
    # pins = [top_pins, bottom_pins, left_pins, right_pins]
    side_names = ["top", "bottom", "left", "right"]

    # Collect pin data per side
    side_pins = {s: [] for s in side_names}
    for side_idx, side in enumerate(side_names):
        for i, (px, py, pw, ph) in enumerate(pins[side_idx]):
            side_pins[side].append(
                {
                    "side": side,
                    "local_idx": i,
                    "px": px,
                    "py": py,
                    "w": pw,
                    "h": ph,
                }
            )

    # Compute per-side trimmed means for width/height
    side_stats = {}
    for side, plist in side_pins.items():
        if not plist:
            continue

        widths = np.array([p["w"] for p in plist], dtype=float)
        heights = np.array([p["h"] for p in plist], dtype=float)

        med_w = np.median(widths)
        med_h = np.median(heights)

        filt_w = widths[(widths > 0.5) & (widths <= 1.5 * med_w)]
        filt_h = heights[(heights > 0.5) & (heights <= 1.5 * med_h)]

        mean_w = filt_w.mean() if len(filt_w) else med_w
        mean_h = filt_h.mean() if len(filt_h) else med_h

        side_stats[side] = (mean_w, mean_h)

    # Mark outliers
    width_rel_thresh = 0.12
    height_rel_thresh = 0.07
    outlier_ids = set()

    for side, plist in side_pins.items():
        if side not in side_stats:
            continue
        med_w, med_h = side_stats[side]
        if med_w <= 0 or med_h <= 0:
            continue

        for p in plist:
            dw = abs(p["w"] - med_w) / med_w
            dh = abs(p["h"] - med_h) / med_h

            if side in ("top", "bottom"):
                if dw > width_rel_thresh or dh > height_rel_thresh:
                    outlier_ids.add((p["side"], p["local_idx"]))
            else:  # left, right
                if dw > height_rel_thresh or dh > width_rel_thresh:
                    outlier_ids.add((p["side"], p["local_idx"]))

    print("Outlier pins (side, idx):", sorted(outlier_ids))

    # Draw rotated rectangles on img
    output = img
    base_color = (0, 200, 0)
    outlier_color = (0, 0, 255)

    h, w = output.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    def rotate_point(x, y, M):
        v = np.array([x, y, 1.0], dtype=float)
        xr, yr = M.dot(v)
        return int(round(xr)), int(round(yr))

    expand = 40

    for side_idx, side in enumerate(side_names):
        for local_idx, (px, py, pw, ph) in enumerate(pins[side_idx]):
            key = (side, local_idx)
            color = outlier_color if key in outlier_ids else base_color

            if side == "top":
                corners = [
                    (px,         py),
                    (px + pw,    py),
                    (px + pw,    py + ph + expand),
                    (px,         py + ph + expand),
                ]
            elif side == "bottom":
                corners = [
                    (px,         py - expand),
                    (px + pw,    py - expand),
                    (px + pw,    py + ph),
                    (px,         py + ph),
                ]
            elif side == "right":
                corners = [
                    (px - expand, py),
                    (px + pw,     py),
                    (px + pw,     py + ph),
                    (px - expand, py + ph),
                ]
            else:  # left
                corners = [
                    (px,              py),
                    (px + pw + expand, py),
                    (px + pw + expand, py + ph),
                    (px,              py + ph),
                ]

            rot_corners = [rotate_point(x, y, M) for (x, y) in corners]
            pts = np.array(rot_corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], isClosed=True, color=color, thickness=6)

    return output


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Detect chip pins and highlight defects.')
    parser.add_argument('input', help='Path to input PNG image (chip image).')
    parser.add_argument('--output', default='final/00_output.png', help='Path for output image.')
    args = parser.parse_args()

    # Make sure output directory exists if there is one
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError("Could not read input image")
    # Adjust thresh_value here (0–255)
    img_for_defects = cv2.imread(args.input)
    img1 = create_bw_image(args.input)

    img2_body = make_bw(gray, thresh_value=120)
    img2_strict = make_bw(gray, thresh_value=243)  
    img2_loose = make_bw(gray, thresh_value=220)  

    img3_pre = fill_largest_white(img2_strict)    
    img3 = fill_largest_white(img2_loose)   
    
    img4_1 = merge_black_regions(img2_body, img1)
    img4_2 = merge_black_regions(img3_pre, img4_1)
    img4 = merge_black_regions(img3, img4_2)   
    
    img_ICbbox, x, y, w, h, = find_chip_with_pins_bbox(img1)
    chip_x, chip_y, chip_w, chip_h = x, y, w+10, h+10  
    img5_clean = fill_outside_bbox(img4, chip_x, chip_y, chip_w, chip_h)
    
    img6_morph = fill_white_inside_black(img5_clean, kx=7, ky=7, iterations=1)
    img7_fill_small_white = fill_small_white_areas(img6_morph, max_white_area=10000)
    
    img8_aligned, angle = rotate_chip_to_horizontal(img1, img7_fill_small_white)

    chip_x, chip_y, chip_w, chip_h = compute_bbox_body(img8_aligned)

    
    gray = img8_aligned
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary_for_scan = cv2.bitwise_not(binary)  # now chip+pins black (0), background white (255)

    pins = detect_pins_by_scanning(
        binary_for_scan,
        chip_x, chip_y, chip_w, chip_h,
        scan_offset=5,          # start a bit away from body
        scan_shift_to_edge=0,   # how far earlier to start scanning
        min_pin_thickness=20,    # ignore tiny noise  
        margin=5,
        min_box_w=35,
        min_box_h=80,
        min_box_area=2000)
    
    
    img_for_defects = cv2.imread(args.input)
    output_img = detect_defect_pins(pins, img_for_defects, angle)
    cv2.imwrite(args.output, output_img)    

