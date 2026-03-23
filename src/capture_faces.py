import argparse
import os

import cv2
try:
    from .face_utils import (
        TARGET_SIZE,
        crop_with_padding,
        detect_primary_face,
        load_face_cascades,
        prepare_for_detection,
    )
except ImportError:
    from face_utils import (
        TARGET_SIZE,
        crop_with_padding,
        detect_primary_face,
        load_face_cascades,
        prepare_for_detection,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture face images from webcam and add them to the dataset."
    )
    parser.add_argument("--name", required=True, help="Folder name / label to save under Dataset")
    parser.add_argument("--count", type=int, default=20, help="Number of images to collect")
    return parser.parse_args()


def ensure_output_dir(person_name):
    output_dir = os.path.join("Dataset", person_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def next_index(output_dir):
    existing = []
    for file_name in os.listdir(output_dir):
        stem, ext = os.path.splitext(file_name)
        if ext.lower() == ".pgm" and stem.isdigit():
            existing.append(int(stem))

    return max(existing, default=0) + 1

def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.name)
    image_index = next_index(output_dir)
    saved_count = 0

    cap = cv2.VideoCapture(0)
    cascades = load_face_cascades()

    print(f"Saving images to: {output_dir}")
    print("Controls: press S to save the detected face, ESC to quit.")
    print("Tip: turn your head slowly left/right until the green box follows your face.")

    while saved_count < args.count:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_gray = prepare_for_detection(frame)

        display = frame.copy()
        status = f"Saved: {saved_count}/{args.count}"
        face_info = detect_primary_face(detect_gray, cascades)

        if face_info is not None:
            x, y, w, h, detector_name = face_info
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_crop = crop_with_padding(gray, x, y, w, h)
            status = f"{status} | {detector_name}"
        else:
            face_crop = None
            status = f"{status} | No face detected"

        cv2.putText(
            display,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Capture Faces", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == ord("s") and face_crop is not None:
            face_crop = cv2.resize(face_crop, TARGET_SIZE)
            file_path = os.path.join(output_dir, f"{image_index}.pgm")
            cv2.imwrite(file_path, face_crop)
            print(f"Saved {file_path}")
            image_index += 1
            saved_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Collected {saved_count} image(s) for {args.name}.")


if __name__ == "__main__":
    main()
