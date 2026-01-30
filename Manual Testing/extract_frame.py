import cv2
import os
import random

video_path = "Patterns.mp4"
output_dir = "validation_frames"
num_frames_to_extract = 20   # change to 30 or 50 if needed

os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total_frames)

# Randomly choose frame indices
frame_indices = sorted(random.sample(range(total_frames), num_frames_to_extract))
print("Extracting frames:", frame_indices)

count = 0
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frame_path = os.path.join(output_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(frame_path, frame)
        print("Saved:", frame_path)
        count += 1

cap.release()
print(f"Done. Extracted {count} frames.")