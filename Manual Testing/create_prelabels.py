# create_prelabels.py
import os, glob, cv2, numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

# -----------------
# USER TUNEABLES
# -----------------
IMG_DIR = "validation_frames"   # folder with frame_XXXXX.png
MIN_AREA = 8
MAX_AREA = 120
BRIGHTNESS_OFFSET = -5
LABEL_NAME = "firefly"          # class name in XML
# -----------------

def detect_fireflies(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, BRIGHTNESS_OFFSET)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")

def write_voc_xml(filename_img, boxes, label=LABEL_NAME):
    img = cv2.imread(filename_img)
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(filename_img))[0]
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = os.path.basename(IMG_DIR)
    ET.SubElement(root, 'filename').text = os.path.basename(filename_img)
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(w)
    ET.SubElement(size, 'height').text = str(h)
    ET.SubElement(size, 'depth').text = str(3)
    for (x,y,bbw,bbh) in boxes:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x)
        ET.SubElement(bndbox, 'ymin').text = str(y)
        ET.SubElement(bndbox, 'xmax').text = str(x + bbw)
        ET.SubElement(bndbox, 'ymax').text = str(y + bbh)
    xml_str = prettify_xml(root)
    xml_path = os.path.join(os.path.dirname(filename_img), base + '.xml')
    with open(xml_path, 'w') as f:
        f.write(xml_str)

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")) + glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print("Found", len(files), "images")
    for i, img_path in enumerate(files):
        img = cv2.imread(img_path)
        boxes = detect_fireflies(img)
        write_voc_xml(img_path, boxes)
        # optional: draw boxes on image for quick checking
        vis = img.copy()
        for (x,y,w,h) in boxes:
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 1)
        cv2.imwrite(os.path.join(IMG_DIR, "pred_"+os.path.basename(img_path)), vis)
        if i % 5 == 0:
            print(f"Processed {i}/{len(files)}")
    print("Prelabels created. Open validation_frames/ in LabelImg to correct.")