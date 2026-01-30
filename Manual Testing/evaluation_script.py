import json
import numpy as np
import glob
import os

GT_DIR = "validation_points"
PRED_DIR = "predicted_points"
MAX_DIST = 30  # accepted distance for correct match (you can tune)

def load_points(path):
    with open(path, "r") as f:
        return json.load(f)

def compare_points(gt_points, pred_points, max_dist=MAX_DIST):
    gt_used = set()
    TP = FP = FN = 0

    for px, py in pred_points:
        best = None
        best_d = max_dist

        for idx, (gx, gy) in enumerate(gt_points):
            d = np.hypot(px - gx, py - gy)
            if d < best_d:
                best_d = d
                best = idx

        if best is not None:
            TP += 1
            gt_used.add(best)
        else:
            FP += 1

    FN = len(gt_points) - len(gt_used)
    return TP, FP, FN

# ---- Loop through all frames ----

gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.json")))

total_TP = total_FP = total_FN = 0

for gt_file in gt_files:
    base = os.path.basename(gt_file)
    pred_file = os.path.join(PRED_DIR, base)

    if not os.path.exists(pred_file):
        print("Missing prediction for:", base)
        continue

    gt_points = load_points(gt_file)
    pred_points = load_points(pred_file)

    TP, FP, FN = compare_points(gt_points, pred_points)

    total_TP += TP
    total_FP += FP
    total_FN += FN

    print(f"{base}:  TP={TP},  FP={FP},  FN={FN}")

precision = total_TP / (total_TP + total_FP + 1e-9)
recall = total_TP / (total_TP + total_FN + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("\n===== SUMMARY =====")
print("Total TP:", total_TP)
print("Total FP:", total_FP)
print("Total FN:", total_FN)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")