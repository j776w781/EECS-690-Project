import cv2
import numpy as np

# This function it produces a stable, noise-reduced binary image that makes 
# pin detection easier, but alone not good enough, needs more preprocessing 
def create_bw_image(img):
    # Technique used -> CLAHE
    # To improve visibility of details in regions with low contrast
    im1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

    # Technique used -> non-local means denoising
    # To remove sensor noise and small grain patterns without destroying edges
    im2 = cv2.fastNlMeansDenoising(im1, None, h=14, templateWindowSize=7, searchWindowSize=21)

    # Technique used -> bilateral filtering
    #    smooths noise while preserving edges (unlike a normal blur)
    im2 = cv2.bilateralFilter(im2, d=5, sigmaColor=135, sigmaSpace=20)

    # Technique used -> Gaussian blur + unsharp masking
    # To soften the image slightly and then combine blurred and original versions
    # to enhance edges and make thresholds more reliable
    blur = cv2.GaussianBlur(im2, (3, 3), 0)
    im3 = cv2.addWeighted(im2, 1.4, blur, -0.4, 0)

    # Technique used -> adaptive thresholding (gaussian)
    # To convert the image to black/white while adapting to local lighting
    adaptive_bw = cv2.adaptiveThreshold(
        im3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5)

    return adaptive_bw

# This function converts a image to a binary image with a given parameter,
# to be able to isloate the pins parts from the background 
def make_bw(img, thresh_value):

    _, bw = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    return bw

# This function removes the largest white area of the image.
# Because the the input has a high threshold value for binary image 
# this then isolates the parts of definite pins we can merge with the lower threshold image 
# to get a more accurate result
def fill_largest_white(img):

    # Create a mask of white pixels
    white_mask = (img == 255).astype(np.uint8)

    # Connected components to find white regions
    #    connectivity=8: diagonals are connected too
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

    # label 0 is background; skip it
    if num_labels <= 1:
        return img

    # Find the largest white component (by area)
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_label = 1 + np.argmax(areas)  # shift back by 1

    # 6) Fill that component black in the original image
    result = img.copy()
    result[labels == largest_label] = 0  # set to black
    final = 255 - result

    return final


# Merge the balck regions of image_1 with image_2.
def merge_black_regions(image_1, image_2):
    merged = image_2.copy()
    merged[image_1 == 0] = 0

    return merged

# This function finds makes bounding box around the entire chip.
# With help of this boundingbox we can remove all noise outside the bbox 
# more radically, but reliably.
def find_chip_with_pins_bbox(img):
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)

    # Techique used -> Morphological dilation:
    # To thicken thin pins and merge small broken pin fragments
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, dil_k, iterations=2)

    # Technique used -> Morphological closing:
    # To close tiny gaps between pin segments and smooth out the pin/chip region 
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)

    # Technique used: Morphological opening:
    # To remove tiny noise specks for more accurate/reliable result
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=1)

    # Technique used -> Contour detection
    # To find connected white regions (chip body + pins)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found! Line 85")

    # Select the largest contour
    #  -> Now chip + pins assembly is the largest object
    def bbox_area(c):
        x, y, w, h = cv2.boundingRect(c)
        return w * h

    cnt = max(contours, key=bbox_area)

    # Compute the smallest convex shape that encloses the contour.
    # To remove concave irregularities by gaps between pins 
    hull = cv2.convexHull(cnt)

    x, y, w, h = cv2.boundingRect(hull)
    out = img.copy()

    return out, x, y, w, h

# This function keeps only the chip region inside a bounding box (the chip body+pins) 
# and turns everything outside that region completely white.
def fill_outside_bbox(img, bbox_x, bbox_y, bbox_w, bbox_h):

    # Create an all-white image
    cleaned = np.full_like(img, 255)

    # Restore only the requested bounding box region
    x0, y0, w, h = bbox_x, bbox_y, bbox_w, bbox_h
    cleaned[y0:y0+h, x0:x0+w] = img[y0:y0+h, x0:x0+w]

    return cleaned

# This function reomves noise by using morphological closing inside the bounding box of the entire chip
def fill_white_inside_black(img, kx, ky, iterations):
    # Invert: chip+pins = white, background = black
    inv = 255 - img

    # Technique used -> Morphological closing:
    # Fill gaps/holes in chip+pins
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    inv_closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel,
                                  iterations=iterations)

    # Everything that is white in inv_closed is "chip region"
    chip_mask = inv_closed == 255

    # In the ORIGINAL image, force that region to black
    result = img.copy()
    result[chip_mask] = 0

    return result

# This function fills in the small white areas that is just noise  
def fill_small_white_areas(img, max_white_area):

    # 2) Create mask where white = 1
    white_mask = (img == 255).astype(np.uint8)

    # Connected components on white regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

    result = img.copy()

    # Fill small white components to black
    for label in range(1, num_labels):  
        area = stats[label, cv2.CC_STAT_AREA]
        # Remove area if it is not large enough
        if area < max_white_area:
            result[labels == label] = 0

    return result

# This function computes the angle the chip is rotated/angled in the input image.
# For a reliable outcome a chip that is parallel to the image border is essential
def rotate_chip_to_horizontal(img, img7_fill_small_white):

    # Prepare the image for a reliable angle computation 
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the largest contour
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chip_cnt = max(contours, key=cv2.contourArea)

    _, (w, h), angle = cv2.minAreaRect(chip_cnt)

    # Check if angle refers to long side
    if w < h:
        angle += 90.0

    # The maximum angle we really care about is smaller than 90, beause we know it is an rectangle
    # Therefore, normalize it.
    while angle >= 90.0:
        if (angle - 90.0) < 0:
            break
        angle -= 90.0

    if angle < 0:
        angle = 0

    # Safety precaution
    if angle > 10:
        H, W = img.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1.0)
        img = img7_fill_small_white
        rotated = cv2.warpAffine(
            img, M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        angle = 0
    else:
        # Rotate the image by the normalized angle 
        H, W = img.shape[:2]
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img = img7_fill_small_white
        rotated = cv2.warpAffine(
            img, M, (W, H),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        # return rotated image and angle because we need this angle to rotate the pins bbox back
    return rotated, angle

# Calculate how many transitions a region is by counting how often pixel values switch between black and white
# A lot of transistions would mean that pins are still inside the bounding box which we do not want to include
def calculate_frequency_switches(region):
    if region.size == 0:
        return float('inf')
    
    # Count horizontal black/white transitions 
    h_diff = np.abs(np.diff(region.astype(int), axis=1))
    h_switches = np.sum(h_diff > 127)
    
    # Count horizontal black/white transitions
    v_diff = np.abs(np.diff(region.astype(int), axis=0))
    v_switches = np.sum(v_diff > 127)
    
    # Normalize by area and return the "frequency" so how many transistion the current border would have
    # Goal is it to have a very low frequency
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
    
    # Calculate the ratio of black pixels of the current border
    border_pixels = np.concatenate([top_border, bottom_border, left_border, right_border])
    black_count = np.sum(border_pixels < 127)
    total_border_pixels = len(border_pixels)
    
    return black_count / total_border_pixels if total_border_pixels > 0 else 0

def compute_bbox_body(gray_img):

    # Techniques used -> Thresholding and morphological erosion: 
    # Convert the chip image to a binary mask and heavily erode it so that 
    # thin structures like pins disappear while the main chip body remains. 
    _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    eroded = cv2.erode(binary, np.ones((15, 15), np.uint8), iterations=2)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Techniques used -> Contour detection and scoring: 
    # Find all large contours and evaluate each one based on border blackness 
    # and internal uniformity, keeping the contour most likely to be the chip 
    # body. 
    chip_x = chip_y = chip_w = chip_h = 0
    best_score = -1
    for c in contours:
        if cv2.contourArea(c) < 10000: continue
        x, y, w, h = cv2.boundingRect(c)
        region = gray_img[y:y+h, x:x+w]
        s = calculate_border_black_ratio(region) * 0.6 + (1/(1+calculate_frequency_switches(region)*100)) * 0.4
        if s > best_score:
            best_score = s
            chip_x, chip_y, chip_w, chip_h = x, y, w, h
    if best_score == -1: exit()

    # Bounding box expansion to ensure the entire chip body is included
    chip_x = max(0, chip_x - 10)
    chip_y = max(0, chip_y - 10)
    chip_w = min(gray_img.shape[1] - chip_x, chip_w + 20)
    chip_h = min(gray_img.shape[0] - chip_y, chip_h + 20)

    # Iterative border refinement: Because the previous techniques are not reliable enough we need
    # to refine the broder fruther. So we repeatedly analyze the outer border of the region
    # and shrink the bounding box on sides containing too many white pixels, which
    # means there is still background or pins included. This shrinks the box to focus on the
    # solid central chip. 
    for _ in range(1000):
        region = gray_img[chip_y:chip_y+chip_h, chip_x:chip_x+chip_w]
        br = calculate_border_black_ratio(region)
        fr = calculate_frequency_switches(region)
        if br >= 0.85 and fr <= 0.02: break
        h, w = region.shape
        if h < 20 or w < 20: break
        t = np.sum(region[0] > 127)
        b = np.sum(region[-1] > 127)
        l = np.sum(region[:,0] > 127)
        r = np.sum(region[:,-1] > 127)
        shr = False
        if br < 0.85:
            if t > w*0.2 and chip_h > 20: chip_y += 10; chip_h -= 10; shr = True
            if b > w*0.2 and chip_h > 20: chip_h -= 10; shr = True
            if l > h*0.2 and chip_w > 20: chip_x += 10; chip_w -= 10; shr = True
            if r > h*0.2 and chip_w > 20: chip_w -= 10; shr = True
        elif fr > 0.02 and chip_h > 20 and chip_w > 20:
            chip_y += 10; chip_h -= 20
            chip_x += 10; chip_w -= 20
            shr = True
        if not shr: break

    # Techniques used -> Morphological closing and opening: 
    # Here we clean the binary image by removing small noise and closing gaps
    k = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, k, iterations=1)

    return chip_x, chip_y, chip_w, chip_h

def detect_pins_by_scanning(binary, chip_x, chip_y, chip_w, chip_h,
                            scan_offset, min_pin_thickness, margin, 
                            min_box_w, min_box_h, min_box_area, disconnected_tolerance):

    H, W = binary.shape
    PIN = 0  

    # Second offset if noise near the chip body is to much and second iteration further away is needed
    SCAN_OFFSET_REVISED = 50  

    # Forces a coordinates to stay within a minimum–maximum range
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # In the function we scan over the line/bounding box of the chip and find
    # continuous black segments (indicating pins)
    # Arguments:    - get_pixel_line: the corresponding border (t, b, l ,r)
    #               - coord_range:    the range of this border (t, b, l ,r)
    def scan_edge(get_pixel_line, coord_range):
        # hits saves the found pins on that chip border in a list
        hits = [] 
        for coord in range(*coord_range):
            d = 0
            inside = False
            black_start = None
            while True:
                # get_pixel_top       -> read upward
                # get_pixel_bottom    -> read downward
                # get_pixel_left      -> read left
                # get_pixel_right     -> read right
                p = get_pixel_line(coord, d)
                
                if p is None:
                    if inside:
                        if d - black_start >= min_pin_thickness:
                            hits.append((coord, black_start, d))
                    break
                
                # If pixel "p" is black (PIN = 0) means we are beginning a pin -> update bools
                # When it sees white (background) means we are ending a pin -> update bools
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


    # Merge nearby/neighboring hits into one single "hit"
    # Pins are not just one pixel thick, with a 2 pixel tolerance 
    # (if 2 pixel are white-> still recognized as a single hit/pin)
    #
    # ==> MERGE ALONG CHIP BODY
    #
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

    # Due to reflection in the original image some pins have a white spot in the middle,
    # which leads to a white area after preprocessing to binary image
    # This function connects the pin back together by looking if more black (pin) is inside 
    # a certain tolerance.
    #
    #       (pin)      (pin)
    #        \/         \/
    #
    #       ||||       ||||
    #       ||||       ||||
    #       ||||              (white spot -> only lower part would be boxed in, but with this we search for more
    #       ||||       ||||     black region above the white area, which then also recognizes the top part as pin)
    #       ||||       ||||
    #       ||||       ||||
    # ////////////////////////////// (-> body)

    def check_disconnected_end(get_pixel_line, coords, current_end, tolerance):
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

   
    top_pins = []
    bottom_pins = []
    left_pins = []
    right_pins = []

    # =====================================================================
    #                             TOP EDGE
    # =====================================================================

    # Return the length of border
    #   - x is the position along the top edge
    #   - yy is the vertical position we want to search

    def get_pixel_top(x, d):
        y_start = chip_y - scan_offset 
        yy = y_start - d
        if yy < 0:
            return None
        return binary[yy, x]
    
    # find hits/pins on top border and then groups them
    top_hits = scan_edge(get_pixel_top, (chip_x, chip_x + chip_w))
    top_groups = group_hits(top_hits)


    # For each pin group build a bounding box
    for coords, starts, ends in top_groups:
        # Compute pin width along x (along the border)
        x1 = min(coords) - 3  
        x2 = max(coords) + 3  

        # Compute pin length d of pin (away from the chip body)
        d_start_eff = max(0, min(starts) )
        d_end_eff   = max(d_start_eff, max(ends) )

        # Check for disconnected pin end (see *check_disconnected_end comment* )
        d_end_extended = check_disconnected_end(get_pixel_top, coords, d_end_eff, disconnected_tolerance)
        
        # Convert length of pin "d" into actual image y-coordinates
        y_inner = chip_y
        y_outer = chip_y - d_end_extended - margin
        y_outer -= scan_offset
        y_outer = clamp(y_outer, 0, chip_y)

        # Fix order of y if wrong
        if y_inner > y_outer:
            x1 = clamp(x1, 0, W - 1) 
            x2 = clamp(x2, 0, W - 1)
            w = x2 - x1 + 1
            h = y_inner - y_outer + 1
            area = w * h

        # Because of the images sometimes shade of the chip body/pins make black regions in the binary image between pins.
        # This leads to the problem that some pins are merges as one pin. This part we scan again but further away from 
        # the chip body border. Namely "SCAN_OFFSET_REVISED" further away. 
        # If the width is larger than 120 we are pretty sure that at least two pins are merged into one.
        if w > 120:
            revised_pins = []

            # Return the length of oversized pin boundry box
            #   - x is the position along the top edge
            #   - yy is the vertical position we want to search
            def get_pixel_top_rev(x, d):
                y_start = chip_y - SCAN_OFFSET_REVISED 
                yy = y_start - d
                if yy < 0:
                    return None
                return binary[yy, x]

            # Rescan only inside oversized region
            sub_hits = scan_edge(get_pixel_top_rev, (x1, x2 + 1))
            sub_groups = group_hits(sub_hits)

            # For each pin group build a bounding box
            for s_coords, s_starts, s_ends in sub_groups:
                # Compute pin width along x (along the border)
                sd_start_eff = max(0, min(s_starts) )
                sd_end_eff   = max(sd_start_eff, max(s_ends) )

    
                # Check for disconnected pin end (see *check_disconnected_end comment* )
                sd_end_extended = check_disconnected_end(get_pixel_top_rev, s_coords, sd_end_eff, disconnected_tolerance)

                # Do same calculations of before but with a higher scan offset
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

                # Check if minimum requirement for box size is fullfilled
                if sw < min_box_w or sh < min_box_h or sarea < min_box_area:
                    # fall back to original big box
                    top_pins.append((x1, y_outer, w, h))
                    revised_pins = []
                    break

                revised_pins.append((sx1, y_outer2, sw, sh))

            # Add the new pins to the original pin list
            top_pins.extend(revised_pins)
            continue
        
        # Check if minimum requirement for box size is fullfilled
        if w < min_box_w or h < min_box_h or area < min_box_area:
            continue

        top_pins.append((x1, y_outer, w, h))

    # =====================================================================
    #                           BOTTOM EDGE
    # =====================================================================
    #
    #   Same as TOP edge just that we scan downward
    #   
    def get_pixel_bottom(x, d):
        y_start = chip_y + chip_h + scan_offset
        yy = y_start + d
        if yy >= H:
            return None
        return binary[yy, x]

    bottom_hits = scan_edge(get_pixel_bottom, (chip_x, chip_x + chip_w))
    bottom_groups = group_hits(bottom_hits)

    for coords, starts, ends in bottom_groups:
        d_start_eff = max(0, min(starts) )
        d_end_eff   = max(d_start_eff, max(ends) )

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
                y_start = chip_y + chip_h + SCAN_OFFSET_REVISED
                yy = y_start + d
                if yy >= H:
                    return None
                return binary[yy, x]

            sub_hits = scan_edge(get_pixel_bottom_rev, (x1, x2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) )
                sd_end_eff   = max(sd_start_eff, max(s_ends) )

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
    #                           LEFT EDGE 
    # =====================================================================
    #
    # Same as other edges just that we scan to the left
    #   
    def get_pixel_left(y, d):
        x_start = chip_x - scan_offset 
        xx = x_start - d
        if xx < 0:
            return None
        return binary[y, xx]

    left_hits = scan_edge(get_pixel_left, (chip_y, chip_y + chip_h))
    left_groups = group_hits(left_hits)

    for coords, starts, ends in left_groups:
        y1 = min(coords) - 3  # extend height
        y2 = max(coords) + 3  # extend height

        d_start_eff = max(0, min(starts) )
        d_end_eff   = max(d_start_eff, max(ends) )

        # Check for disconnected pin end
        d_end_extended = check_disconnected_end(get_pixel_left, coords, d_end_eff, disconnected_tolerance)

        x_inner = chip_x
        x_outer = chip_x - d_end_extended - margin
        x_outer -= scan_offset
        x_outer = clamp(x_outer, 0, chip_x)

        if x_inner > x_outer:
            y1 = clamp(y1, 0, H - 1) 
            y2 = clamp(y2, 0, H - 1)
            w = x_inner - x_outer + 1
            h = y2 - y1 + 1
            area = w * h

        if h > 120:
            revised_pins = []

            def get_pixel_left_rev(y, d):
                x_start = chip_x - SCAN_OFFSET_REVISED - scan_offset 
                xx = x_start - d
                if xx < 0:
                    return None
                return binary[y, xx]

            sub_hits = scan_edge(get_pixel_left_rev, (y1, y2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) )
                sd_end_eff   = max(sd_start_eff, max(s_ends) )

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
    #                           RIGHT EDGE
    # =====================================================================
    #
    # Same as other edges just that we scan to the right
    #   
    def get_pixel_right(y, d):
        x_start = chip_x + chip_w + scan_offset
        xx = x_start + d
        if xx >= W:
            return None
        return binary[y, xx]

    right_hits = scan_edge(get_pixel_right, (chip_y, chip_y + chip_h))
    right_groups = group_hits(right_hits)

    for coords, starts, ends in right_groups:
        d_start_eff = max(0, min(starts) )
        d_end_eff   = max(d_start_eff, max(ends) )

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
                x_start = chip_x + chip_w + SCAN_OFFSET_REVISED + scan_offset
                xx = x_start + d
                if xx >= W:
                    return None
                return binary[y, xx]

            sub_hits = scan_edge(get_pixel_right_rev, (y1, y2 + 1))
            sub_groups = group_hits(sub_hits)

            for s_coords, s_starts, s_ends in sub_groups:
                sd_start_eff = max(0, min(s_starts) )
                sd_end_eff   = max(sd_start_eff, max(s_ends) )

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



    # This is a filter that removes all side pins if there are less than 9 pins/boxes per side
    # Because if so it is very likely that that side does not have any pin
    pins = []
    pins.append(top_pins if len(top_pins) >= 4 else [])
    pins.append(bottom_pins if len(bottom_pins) >= 4 else [])
    pins.append(left_pins if len(left_pins) >= 4 else [])
    pins.append(right_pins if len(right_pins) >= 4 else [])

    return pins

def detect_defect_pins(pins, img, angle):

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

        # Compute a average width and height of a pin per side.
        # In this average the pins that are definite outliers are NOT included
        widths = np.array([p["w"] for p in plist], dtype=float)
        heights = np.array([p["h"] for p in plist], dtype=float)

        med_w = np.median(widths)
        med_h = np.median(heights)

        filt_w = widths[(widths > 0.5) & (widths <= 1.5 * med_w)]
        filt_h = heights[(heights > 0.5) & (heights <= 1.5 * med_h)]

        mean_w = filt_w.mean() if len(filt_w) else med_w
        
        mean_h = filt_h.mean() if len(filt_h) else med_h

        side_stats[side] = (mean_w, mean_h)

    # Setting the outliers treshholds
    # width has a higher threshhold, due to more noise near the body of the chip, where we have a lot of noise
    # height is more strict
    width_rel_thresh = 0.12
    height_rel_thresh = 0.07
    outlier_ids = set()

    # This checks if the pin is an outlier or not based on the treshholds
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

    print("Found defect pins are (side, idx):", sorted(outlier_ids))

    # Set colors of the normal pins and defect pins
    base_color = (0, 200, 0)
    outlier_color = (0, 0, 255)

    # Calculate the rotationmatrix, to align the box we project on the original image  
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    def rotate_point(x, y, M):
        v = np.array([x, y, 1.0], dtype=float)
        xr, yr = M.dot(v)
        return int(round(xr)), int(round(yr))

    expand = 40

    # Add the boxes into the original image
    for side_idx, side in enumerate(side_names):
        for local_idx, (px, py, pw, ph) in enumerate(pins[side_idx]):

            # Set color
            pin = (side, local_idx)
            color = outlier_color if pin in outlier_ids else base_color

            # Make the bounding boxes around the pins a bit longer towards the chip body 
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

            # Rotate the the box so it aligns with the pin in the original image 
            rot_corners = [rotate_point(x, y, M) for (x, y) in corners]
            pts = np.array(rot_corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=6)

    return img


def mark_pins(input_file):
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(input_file)
    
    img1 = create_bw_image(img)

    img2_body = make_bw(img, 120)
    img2_strict = make_bw(img, 243)  
    img2_loose = make_bw(img, 220)  

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
    binary_for_scan = cv2.bitwise_not(binary)
     

    pins = detect_pins_by_scanning(
        binary_for_scan,
        chip_x, chip_y, chip_w, chip_h,     # Chip body boundry box
        scan_offset=5,                      # Scan offset from body outward in pixel             #
        min_pin_thickness=20,               # Minimum thickness a pin has to have to be recognized as a pin
        margin=5,                           # How far the bounding box should got outward 
        min_box_w=35,                       # Minimum width a pin bounding box needs
        min_box_h=80,                       # Minimum height a pin bounding box needs 
        min_box_area=2000,                  # Minimum area a pin bounding box needs
        disconnected_tolerance=55)          # The tolerance/number of pixels a disconneted pin to be recognized as the same pin along the pin
    
    
    img_for_defects = cv2.imread(input_file)
    output_img = detect_defect_pins(pins, img_for_defects, angle)

    #cv2.imwrite(args.output, output_img)    
    return output_img




