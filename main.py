from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_extraction import apply_pca
from src.train_model import build_reference_stats, train, save_model
from src.evaluate import evaluate
from sklearn.model_selection import train_test_split

# Step 1
images, labels, label_map = load_data("Dataset")

# Step 2
processed = preprocess(images)

# Step 3
X = processed.reshape(len(processed), -1)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train_pca, pca = apply_pca(X_train)
X_test_pca, _ = apply_pca(X_test, pca=pca)

# Step 4
model = train(X_train_pca, y_train)
reference_stats = build_reference_stats(X_train_pca, y_train)

# Step 5
save_model(model, pca, label_map, reference_stats)

# Step 6
evaluate(model, X_test_pca, y_test)
