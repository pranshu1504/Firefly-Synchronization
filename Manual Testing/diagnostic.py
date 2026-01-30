import json, cv2

img = cv2.imread("validation_frames/frame_01501.png")
h, w = img.shape[:2]
print("Original image size:", w, "x", h)

gt = json.load(open("validation_points/frame_01501.json"))
pred = json.load(open("predicted_points/frame_01501.json"))

print("Example GT point:", gt[0])
print("Example Pred point:", pred[0])