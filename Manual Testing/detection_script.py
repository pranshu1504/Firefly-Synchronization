import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

MIN_AREA = 5
MAX_AREA = 80
BRIGHTNESS_OFFSET = -2
INTENSITY_THRESH = 50  # ignore faint flashes


IMG_DIR = "validation_frames"
OUT_DIR = "predicted_points"
os.makedirs(OUT_DIR, exist_ok=True)

def detect_fireflies(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Blur
    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)

    # Step 2: Adaptive threshold
    thresh_adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 21, BRIGHTNESS_OFFSET
    )

    # Step 3: Morphology
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh_adapt, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 4: Contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save detected centroids
    predicted_points = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            # intensity filter
            if img_gray[cy, cx] < INTENSITY_THRESH:
                continue
            predicted_points.append((float(cx), float(cy)))

    return predicted_points


import glob

frames = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))

for f in frames:
    preds = detect_fireflies(f)
    base = os.path.basename(f).replace(".png", "")
    json_path = os.path.join(OUT_DIR, base + ".json")

    with open(json_path, "w") as fp:
        json.dump(preds, fp)

    print("Saved predictions to:", json_path)

print("Done. All predicted points saved.")