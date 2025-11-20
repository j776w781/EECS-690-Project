import cv2
import easyocr
import numpy as np

def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their new adjusted values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def read_ic_text(image_path):
    print("Initializing AI Engine...")
    # 'quantize=False' improves accuracy on small text
    reader = easyocr.Reader(['en'], gpu=False, quantize=False)

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None: return

    # 2. Preprocessing Pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP A: Gamma Correction (The Fix for Faint Etching)
    # Gamma > 1.0 brightens dark/gray areas (the etching) while keeping black black.
    # We use 1.5 to make the gray text pop out.
    gamma_corrected = adjust_gamma(gray, gamma=1.5)

    # STEP B: Aggressive CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # increased clipLimit from 2.0 to 5.0 to force contrast on the faint text
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gamma_corrected)

    # STEP C: Sharpening
    # Laser etching often has soft edges. This kernel sharpens them.
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # STEP D: Invert (White text on Black -> Black text on White)
    inverted = cv2.bitwise_not(sharpened)
    
    # Save debugs to see exactly what the AI sees
    cv2.imwrite("debug_1_gamma.jpg", gamma_corrected)
    cv2.imwrite("debug_2_enhanced.jpg", enhanced)
    cv2.imwrite("debug_3_final_input.jpg", inverted)

    # 3. Run OCR with Low Contrast Mode
    print(f"Reading text...")
    
    # TWEAKS:
    # contrast_ths: Lower this to detect lower contrast text (Default is 0.1)
    # adjust_contrast: Let EasyOCR boost contrast internally too
    results = reader.readtext(inverted, detail=1, 
                              contrast_ths=0.05, 
                              adjust_contrast=0.7,
                              decoder='beamsearch') # Beamsearch is slower but more accurate

    # 4. Visualization
    output_img = img.copy()
    
    print("\n--- Detected Markings ---")
    for (bbox, text, prob) in results:
        # Lower confidence threshold slightly (0.3) because etching is hard
        if prob > 0.3: 
            print(f"Found: '{text}' (Confidence: {prob:.1%})")
            
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))

            cv2.rectangle(output_img, tl, br, (0, 255, 0), 2)
            cv2.putText(output_img, text, (tl[0], tl[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("result_text_enhanced.jpg", output_img)
    print("Success! Check 'result_text_enhanced.jpg'")

if __name__ == "__main__":
    read_ic_text("IC marking images/A-J-28SOP-03F-SM.png")