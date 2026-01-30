import numpy as np
import json, glob, os, cv2
import matplotlib.pyplot as plt

GT_DIR = "validation_points"
PRED_DIR = "predicted_points"

distances = []

for gt_file in sorted(glob.glob(os.path.join(GT_DIR, "*.json")))[:10]:  # check first 10 frames
    base = os.path.basename(gt_file)
    pred_file = os.path.join(PRED_DIR, base)
    if not os.path.exists(pred_file): 
        continue

    gt = json.load(open(gt_file))
    pred = json.load(open(pred_file))

    # compute distances from each pred to nearest gt
    for px, py in pred:
        dists = [np.hypot(px-gx, py-gy) for gx, gy in gt]
        if dists:
            distances.append(min(dists))

plt.hist(distances, bins=30)
plt.xlabel("Error distance (pixels)")
plt.ylabel("Frequency")
plt.title("GT-Pred centroid mismatch distribution")
plt.show()

print("95th percentile error distance:", np.percentile(distances, 95))
print("99th percentile error:", np.percentile(distances, 99))