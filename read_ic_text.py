import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import cv2
import easyocr


def crop(img):
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    arr = np.array(img)
    # Convert to grayscale
    gray = np.mean(arr, axis=2)

    # Find dark region
    mask = gray < 50

    # Bounding box in ORIGINAL coords
    coords = np.column_stack(np.where(mask))
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    bbox = (xmin, ymin, xmax, ymax)  # (left, top, right, bottom)
    cropped_img = img.crop(bbox)     # PIL Image

    return cropped_img, bbox


def rgb_to_cmyk(image):
    if isinstance(image, Image.Image):
        img = np.asarray(image).astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32) / 255.0

    # Assume img is RGB (from PIL)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    # CMY
    c = 1 - r
    m = 1 - g
    y = 1 - b

    # Key (black)
    k = np.minimum(np.minimum(c, m), y)

    denom = 1 - k
    denom[denom == 0] = 1  # avoid division by zero

    c = (c - k) / denom
    m = (m - k) / denom
    y = (y - k) / denom

    C = (c * 255).astype(np.uint8)
    M = (m * 255).astype(np.uint8)
    Y = (y * 255).astype(np.uint8)
    K = (k * 255).astype(np.uint8)

    return C, M, Y, K


def mult_cmy(C, M, Y):
    C_f = C.astype(np.float32)
    M_f = M.astype(np.float32)
    Y_f = Y.astype(np.float32)

    # mimic your previous logic: invert Y then multiply
    Y_inv = 255.0 - Y_f

    multiplied = np.clip(C_f * M_f * Y_inv, 0, 255).astype(np.uint8)

    kernel = np.ones((7, 7), np.float32) / 25
    gray_blurred = cv2.filter2D(multiplied, -1, kernel)

    return gray_blurred  # uint8 2D


def clean_gray(img):
    _, binary = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    min_area = 250
    mask = np.zeros_like(binary, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == label] = 255

    cleaned = img.copy()
    cleaned[mask == 0] = 255

    return cleaned  # uint8 2D


def multiply_image(img, factor=1.0, target_short_side=256):
    multiplied = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    h, w = multiplied.shape[:2]

    if h < w:
        new_h = target_short_side
        scale = new_h / h
        new_w = int(w * scale)
    else:
        new_w = target_short_side
        scale = new_w / w
        new_h = int(h * scale)

    resized = cv2.resize(multiplied, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized  # uint8 2D



def bbox_to_rect(bbox):
    pts = np.array(bbox, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou(rectA, rectB):
    xA1, yA1, xA2, yA2 = rectA
    xB1, yB1, xB2, yB2 = rectB

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0.0, xA2 - xA1) * max(0.0, yA2 - yA1)
    areaB = max(0.0, xB2 - xB1) * max(0.0, yB2 - yB1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

# Postprocessing
def clean_text(text):
    text.replace("O", "0")
    text.replace("o", "0")
    text.replace("b", "6")
    text.replace("q", "9")
    return text.upper()


def merge_by_iou_keep_best(detections, iou_thresh=0.5):
    clusters = []

    for det in detections:
        match = None
        best_iou = 0.0

        for c in clusters:
            curr_iou = iou(det["rect"], c["rect"])
            if curr_iou >= iou_thresh and curr_iou > best_iou:
                best_iou = curr_iou
                match = c

        if match is None:
            clusters.append(det.copy())
        else:
            if det["conf"] > match["conf"]:
                match.update(det)

            x1 = min(match["rect"][0], det["rect"][0])
            y1 = min(match["rect"][1], det["rect"][1])
            x2 = max(match["rect"][2], det["rect"][2])
            y2 = max(match["rect"][3], det["rect"][3])
            match["rect"] = (x1, y1, x2, y2)

    return clusters


def ocr_on_img(
    gray_cln_shrink: np.ndarray,
    original_path: str,
    crop_bbox,
    iou_thresh: float = 0.5
):
    if isinstance(original_path, (str, Path)):
        original_path = Path(original_path)
        img_original = cv2.imread(str(original_path))
    else:
        img_original = original_path.copy()
        original_path = None 

    # Shapes
    h_s, w_s = gray_cln_shrink.shape[:2]  # shrinked chip
    # cropped chip size:
    xmin_c, ymin_c, xmax_c, ymax_c = crop_bbox
    crop_w = xmax_c - xmin_c
    crop_h = ymax_c - ymin_c

    # Scale shrinked → cropped chip
    scale_x_crop = crop_w / w_s
    scale_y_crop = crop_h / h_s

    # EasyOCR reader
    reader = easyocr.Reader(['en','af','bs','cs','cy','da','de','es','et','fr','ga','hr','hu',
        'id','is','it','la','lt','lv','mi','ms','mt','nl','no','oc','pl',
        'pt','ro','sk','sl','sq','sv','sw','tl','tr','uz','vi','rs_latin'
    ], gpu=False)


    raw = reader.readtext(
        gray_cln_shrink,
        detail=1,
        canvas_size=1280,
        mag_ratio=1.0
    )

    detections = []

    for (bbox, text, conf) in raw:
        t = clean_text(text)

        # bbox in shrinked back to cropped chip coords
        bbox_crop = [[p[0] * scale_x_crop, p[1] * scale_y_crop] for p in bbox]

        # cropped chip back to original coords via crop bbox offset
        xmin0, ymin0, _, _ = crop_bbox
        bbox_orig = [[p[0] + xmin0, p[1] + ymin0] for p in bbox_crop]
        rect_orig = bbox_to_rect(bbox_orig)

        detections.append({
            "rect": rect_orig,
            "bbox_poly": bbox_orig,
            "text": t,
            "conf": float(conf)
        })

    best = merge_by_iou_keep_best(detections, iou_thresh=iou_thresh)


    annotated = img_original.copy()

    for d in best:
        x1, y1, x2, y2 = map(int, d["rect"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        label = f"{d['text']}"
        text_y = y1 - 4
        if text_y < 10:
            text_y = y1 + 15

        cv2.putText(
            annotated, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,   
            (0, 255, 0), 2,
            cv2.LINE_AA
        )

    print("Detections:")
    for d in best:
        print(d["text"], d["conf"])

    return annotated


def read_ic_text(img):

    # 1) Crop dark chip from original, keep bbox to map back later
    cropped_chip, crop_bbox = crop(img)   

    # 2) Convert cropped chip to CMYK
    C, M, Y, _ = rgb_to_cmyk(cropped_chip)

    # 3) Multiply CMY channels and blur
    gray = mult_cmy(C, M, Y)

    # 4) Remove small noise
    gray_cln = clean_gray(gray)

    # 5) Multiply and resize → gray_cln_shrink
    gray_cln_shrink = multiply_image(gray_cln, factor=1.0)


    # 7) OCR on gray_cln_shrink and annotate ORIGINAL args.input
    annotated_img = ocr_on_img(
        gray_cln_shrink=gray_cln_shrink,
        original_path=img,
        crop_bbox=crop_bbox,
        iou_thresh=0.3
    )

    return annotated_img
