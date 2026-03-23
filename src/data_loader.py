import os
import cv2
import numpy as np
try:
    from .face_utils import TARGET_SIZE
except ImportError:
    from face_utils import TARGET_SIZE


def load_data(dataset_path):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for person in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for img_name in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, TARGET_SIZE)

            images.append(img)
            labels.append(label_id)

        label_id += 1

    return np.array(images), np.array(labels), label_map
