import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

IMG_DIR = "images"
ANN_DIR = "annotations"


X, y = [], []

label_map = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 1  # treat as 'no mask'
}

for file in tqdm(os.listdir(ANN_DIR)):
    if not file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANN_DIR, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_path = os.path.join(IMG_DIR, root.find("filename").text)
    image = cv2.imread(image_path)

    for obj in root.findall("object"):
        label = obj.find("name").text
        if label not in label_map:
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        face = image[ymin:ymax, xmin:xmax]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (100, 100)) / 255.0
        face = np.expand_dims(face, axis=-1)

        X.append(face)
        y.append(label_map[label])

X = np.array(X, dtype="float32")
y = np.array(y, dtype="int")
print(f"Loaded {len(X)} samples.")
np.savez("faces_dataset.npz", X=X, y=y)
