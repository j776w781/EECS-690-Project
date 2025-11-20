import cv2
import easyocr
import numpy as np
import re
import os
import glob

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def isolate_chip_body(image):
    _, thresh = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return image, (0,0)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    padding = 8
    if w > 2*padding and h > 2*padding:
        return image[y+padding:y+h-padding, x+padding:x+w-padding], (x+padding, y+padding)
    return image, (0,0)

def fix_common_ocr_errors(text):
    """
    Fixes 'Leet Speak' errors common in OCR (e.g., Z->2, S->5, O->0)
    But only if the text looks like a Part Number (contains numbers and letters).
    """
    text = text.upper().replace(" ", "")
    
    # Specific Patch for ST Microelectronics
    if text.startswith("STN") or text.startswith("5TM"):
        text = text.replace("STN", "STM").replace("5TM", "STM")
    
    # If it detects 'Stn3zf', the 'z' is likely '2' and 'f' is 'F'
    if "STM32" in text or "STM3" in text:
        text = text.replace("Z", "2")

    # General cleanup for common confusables
    # We only apply this if the text is mixed alpha-numeric
    if re.search(r'[A-Z]', text) and re.search(r'[0-9]', text):
        # B often confused with 8, but 8 is common in part numbers, so be careful.
        # Z often confused with 2
        pass 

    return text

def parse_chip_info(lines):
    info = {
        "Part Number": "Unknown",
        "Date Code": "Unknown",
        "Lot/Trace": "Unknown"
    }
    
    # KNOWN MANUFACTURER PREFIXES
    # If a line starts with these, it is almost certainly the Part Number
    known_prefixes = ["CY", "STM", "ADC", "LM", "AT", "PIC", "TMS", "XC"]

    # Sort lines by vertical position (Top lines first)
    # lines is list of (text, conf)
    # We need index to keep order
    lines_ordered = [(text, conf, i) for i, (text, conf) in enumerate(lines)]

    found_part_no = False

    for text, conf, idx in lines_ordered:
        clean_text = fix_common_ocr_errors(text)
        
        # Ignore noise
        if len(clean_text) < 3: continue

        # --- 1. Part Number Detection ---
        if not found_part_no:
            # Check against known prefixes
            if any(clean_text.startswith(prefix) for prefix in known_prefixes):
                info["Part Number"] = clean_text
                found_part_no = True
                continue
            
            # Heuristic: If it's the TOP line and has letters+numbers, it's likely the part #
            if idx == 0 and re.search(r'[A-Z]', clean_text) and re.search(r'[0-9]', clean_text):
                info["Part Number"] = clean_text
                found_part_no = True
                continue

        # --- 2. Date Code Detection ---
        # Look for 4 digits. 
        # Standard is YYWW. We assume 1990-2030 range (digits 90-99 or 00-30)
        # And Weeks 01-53
        date_match = re.search(r'\b(9\d|0\d|1\d|2\d|30)([0-5]\d)\b', clean_text)
        if date_match and info["Date Code"] == "Unknown":
            info["Date Code"] = date_match.group(0)
            # Don't continue, because date code might be part of Lot string

        # --- 3. Lot/Trace Logic ---
        # If it's not the Part Number, it's likely Lot/Trace
        if clean_text != info["Part Number"]:
            # If we have "ADC" on line 1 and "0832..." on line 2, 
            # we might want to merge them if Part Number is just "ADC"
            if info["Part Number"] in ["ADC", "STM"] and clean_text[0].isdigit():
                 info["Part Number"] += clean_text
                 found_part_no = True
                 continue

            # Otherwise, store as Lot
            if len(clean_text) > len(info["Lot/Trace"]) or info["Lot/Trace"] == "Unknown":
                 # Filter out if it's just the date code repeated
                 if clean_text != info.get("Date Code", ""):
                    info["Lot/Trace"] = clean_text

    return info

def group_text_by_line(results, y_threshold=15):
    valid_results = [r for r in results if r[2] > 0.3]
    if not valid_results: return []

    def get_y_center(bbox): return (bbox[0][1] + bbox[3][1]) / 2
    valid_results.sort(key=lambda x: get_y_center(x[0]))

    lines = []
    current_line = [valid_results[0]]
    
    for i in range(1, len(valid_results)):
        prev_y = get_y_center(current_line[-1][0])
        curr_y = get_y_center(valid_results[i][0])

        if abs(curr_y - prev_y) < y_threshold:
            current_line.append(valid_results[i])
        else:
            lines.append(current_line)
            current_line = [valid_results[i]]
    lines.append(current_line)

    merged_lines = []
    for line in lines:
        line.sort(key=lambda x: x[0][0][0]) 
        full_text = " ".join([item[1] for item in line])
        avg_conf = sum([item[2] for item in line]) / len(line)
        merged_lines.append((full_text, avg_conf))
    return merged_lines

def read_ic_text(image_path, output_filename):
    print(f"Processing: {image_path}")
    reader = easyocr.Reader(['en'], gpu=False, quantize=False)
    
    img = cv2.imread(image_path)
    if img is None: return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped, offset = isolate_chip_body(gray)
    ox, oy = offset

    # V5 Image Processing
    denoised = cv2.fastNlMeansDenoising(cropped, None, h=10, templateWindowSize=7, searchWindowSize=21)
    gamma = adjust_gamma(denoised, gamma=1.5)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gamma)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    final_input = cv2.bitwise_not(sharpened)

    results = reader.readtext(final_input, detail=1, contrast_ths=0.05, adjust_contrast=0.5, decoder='beamsearch')

    print("\n--- Raw Detection ---")
    merged_lines = group_text_by_line(results)
    for line in merged_lines:
        print(f"Line: '{line[0]}' (Conf: {line[1]:.1%})")

    print("\n--- Final Report ---")
    report = parse_chip_info(merged_lines)
    print(f"PART NUMBER : {report['Part Number']}")
    print(f"DATE CODE   : {report['Date Code']}")
    print(f"LOT/TRACE   : {report['Lot/Trace']}")
    print("-" * 30)
    
    output_img = img.copy()
    for (bbox, text, prob) in results:
        if prob > 0.3:
             (tl, tr, br, bl) = bbox
             tl = (int(tl[0]) + ox, int(tl[1]) + oy)
             br = (int(br[0]) + ox, int(br[1]) + oy)
             cv2.rectangle(output_img, tl, br, (0, 255, 0), 1)
    
    cv2.imwrite(output_filename, output_img)
    print(f"Saved image to: {output_filename}\n")

if __name__ == "__main__":
    # Update this path to match your folder
    image_folder = "IC marking images" 
    
    image_files = glob.glob(os.path.join(image_folder, "*.png")) + \
                  glob.glob(os.path.join(image_folder, "*.jpg"))

    print(f"Found {len(image_files)} images.")

    for img_path in image_files:
        # Create a unique filename for each result
        base_name = os.path.basename(img_path)
        output_name = f"result_{base_name}"
        
        read_ic_text(img_path, output_name)