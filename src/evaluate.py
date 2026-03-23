from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True)
    plt.savefig("output/confusion_matrix.png")
    plt.show()