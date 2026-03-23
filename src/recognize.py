import cv2
import joblib
import numpy as np
try:
    from .attendance import mark_attendance
    from .face_utils import (
        TARGET_SIZE,
        crop_with_padding,
        detect_primary_face,
        load_face_cascades,
        prepare_for_detection,
    )
except ImportError:
    from attendance import mark_attendance
    from face_utils import (
        TARGET_SIZE,
        crop_with_padding,
        detect_primary_face,
        load_face_cascades,
        prepare_for_detection,
    )

STABLE_FRAMES_REQUIRED = 4
TRACK_STALE_AFTER = 10


def recognize():
    model = joblib.load("models/svm_model.pkl")
    pca = joblib.load("models/pca_model.pkl")
    label_map = joblib.load("models/labels.pkl")
    reference_stats = joblib.load("models/reference_stats.pkl")
    all_samples = reference_stats["all_samples"]
    all_labels = reference_stats["all_labels"]

    cap = cv2.VideoCapture(0)
    cascades = load_face_cascades()
    stable_name = None
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_gray = prepare_for_detection(frame)
        face_info = detect_primary_face(detect_gray, cascades)

        if face_info is not None:
            x, y, w, h, detector_name = face_info
            face = crop_with_padding(gray, x, y, w, h)

            face = cv2.resize(face, TARGET_SIZE)
            face = cv2.GaussianBlur(face, (5,5), 0)
            face = cv2.equalizeHist(face)

            face = face.reshape(1, -1)
            face_pca = pca.transform(face)

            probabilities = model.predict_proba(face_pca)[0]
            best_index = int(probabilities.argmax())
            confidence = float(probabilities[best_index])
            predicted_label = model.classes_[best_index]
            top_two = sorted(float(score) for score in probabilities)[-2:]
            margin = top_two[-1] - top_two[-2] if len(top_two) == 2 else confidence
            centroid = reference_stats["centroids"][int(predicted_label)]
            centroid_threshold = reference_stats["centroid_thresholds"][int(predicted_label)]
            class_samples = reference_stats["class_samples"][int(predicted_label)]
            nearest_threshold = reference_stats["nearest_thresholds"][int(predicted_label)]
            centroid_distance = float(np.linalg.norm(face_pca[0] - centroid))
            nearest_distance = float(
                np.min(np.linalg.norm(class_samples - face_pca[0], axis=1))
            )
            all_distances = np.linalg.norm(all_samples - face_pca[0], axis=1)
            global_nn_index = int(np.argmin(all_distances))
            global_nn_label = int(all_labels[global_nn_index])
            global_nn_name = label_map[global_nn_label]

            if (
                centroid_distance > centroid_threshold
                and nearest_distance > nearest_threshold
            ):
                candidate_name = "Unknown"
            elif global_nn_label != int(predicted_label):
                candidate_name = "Unknown"
            else:
                candidate_name = label_map[predicted_label]

            if candidate_name == stable_name:
                stable_count += 1
            else:
                stable_name = candidate_name
                stable_count = 1

            if candidate_name == "Unknown":
                display_name = (
                    f"Unknown (p={confidence:.2f}, m={margin:.2f}, "
                    f"dc={centroid_distance:.0f}, dn={nearest_distance:.0f}, "
                    f"nn={global_nn_name})"
                )
                color = (0, 0, 255)
            elif stable_count < STABLE_FRAMES_REQUIRED:
                display_name = (
                    f"Scanning {candidate_name} (p={confidence:.2f}, m={margin:.2f}, "
                    f"dc={centroid_distance:.0f}, dn={nearest_distance:.0f}, "
                    f"nn={global_nn_name})"
                )
                color = (0, 255, 255)
            else:
                display_name = (
                    f"{candidate_name} (p={confidence:.2f}, m={margin:.2f}, "
                    f"dc={centroid_distance:.0f}, dn={nearest_distance:.0f}, "
                    f"nn={global_nn_name})"
                )
                color = (0, 255, 0)
                mark_attendance(candidate_name)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,display_name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            cv2.putText(frame,detector_name,(x,y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        else:
            stable_name = None
            stable_count = 0

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
