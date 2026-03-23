import cv2

TARGET_SIZE = (128, 128)
FACE_PADDING = 0.20


def load_face_cascades():
    cascade_dir = cv2.data.haarcascades
    return [
        ("frontal", cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml"), False),
        ("frontal_alt", cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_alt2.xml"), False),
        ("profile_left", cv2.CascadeClassifier(cascade_dir + "haarcascade_profileface.xml"), False),
        ("profile_right", cv2.CascadeClassifier(cascade_dir + "haarcascade_profileface.xml"), True),
    ]


def prepare_for_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def detect_primary_face(gray, cascades):
    candidates = []

    for detector_name, cascade, flip_image in cascades:
        if cascade.empty():
            continue

        search_img = cv2.flip(gray, 1) if flip_image else gray
        faces = cascade.detectMultiScale(
            search_img,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            if flip_image:
                x = gray.shape[1] - x - w
            candidates.append((x, y, w, h, detector_name))

    if not candidates:
        return None

    return max(candidates, key=lambda face: face[2] * face[3])


def crop_with_padding(gray, x, y, w, h, padding=FACE_PADDING):
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray.shape[1], x + w + pad_x)
    y2 = min(gray.shape[0], y + h + pad_y)
    return gray[y1:y2, x1:x2]
