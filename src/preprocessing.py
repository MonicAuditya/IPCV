import cv2
import numpy as np

def preprocess(images):
    processed = []

    for img in images:
        img = cv2.resize(img, (128, 128))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.equalizeHist(img)
        processed.append(img)

    return np.array(processed)