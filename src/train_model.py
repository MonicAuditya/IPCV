from sklearn.svm import SVC
import joblib
import numpy as np

CENTROID_THRESHOLD_SCALE = 1.10
NEAREST_THRESHOLD_SCALE = 1.25

def train(X_train, y_train):
    model = SVC(kernel='linear', probability=True, class_weight='balanced')
    model.fit(X_train, y_train)

    return model

def build_reference_stats(X_train, y_train):
    centroids = {}
    centroid_thresholds = {}
    class_samples = {}
    nearest_thresholds = {}
    all_samples = []
    all_labels = []

    for label in np.unique(y_train):
        class_vectors = X_train[y_train == label]
        centroid = class_vectors.mean(axis=0)
        distances = np.linalg.norm(class_vectors - centroid, axis=1)
        nearest_distances = []

        for index, vector in enumerate(class_vectors):
            other_vectors = np.delete(class_vectors, index, axis=0)
            if len(other_vectors) == 0:
                nearest_distances.append(0.0)
            else:
                nearest_distances.append(
                    float(np.min(np.linalg.norm(other_vectors - vector, axis=1)))
                )

        centroids[int(label)] = centroid
        centroid_thresholds[int(label)] = float(
            np.max(distances) * CENTROID_THRESHOLD_SCALE
        )
        class_samples[int(label)] = class_vectors
        nearest_thresholds[int(label)] = float(
            max(nearest_distances) * NEAREST_THRESHOLD_SCALE
        )
        all_samples.append(class_vectors)
        all_labels.append(np.full(len(class_vectors), int(label)))

    return {
        "centroids": centroids,
        "centroid_thresholds": centroid_thresholds,
        "class_samples": class_samples,
        "nearest_thresholds": nearest_thresholds,
        "all_samples": np.vstack(all_samples),
        "all_labels": np.concatenate(all_labels),
    }


def save_model(model, pca, label_map, reference_stats):
    joblib.dump(model, "models/svm_model.pkl")
    joblib.dump(pca, "models/pca_model.pkl")
    joblib.dump(label_map, "models/labels.pkl")
    joblib.dump(reference_stats, "models/reference_stats.pkl")
