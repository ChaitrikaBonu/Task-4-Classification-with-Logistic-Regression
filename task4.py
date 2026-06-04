#importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

#loadingDS
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nClass Distribution:")
print(y.value_counts())

#splitting
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

#standardizingFeatures
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)
print("Model Training Completed")

#testing
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#confusionMatrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print(cm)

#precisionAndRecall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#ROC-AUCScore
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

#plottingROCCurve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#trusholdTuning
custom_threshold = 0.3
y_pred_custom = (
    y_prob >= custom_threshold
).astype(int)
print(
    confusion_matrix(
        y_test,
        y_pred_custom
    )
)
print(
    classification_report(
        y_test,
        y_pred_custom
    )
)

#sigmoidFunctionVisualization
x = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-x))
plt.figure(figsize=(8,5))
plt.plot(x, sigmoid)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid(True)
plt.show()
