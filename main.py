"""
Week 4: Machine Learning Model Development and Evaluation
Project: Breast Cancer Classification

Train and compare Logistic Regression and Decision Tree models.
Save evaluation metrics, predictions, and charts in the outputs folder.

This project is for learning purposes only, not medical diagnosis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# 1. Load the dataset
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign

class_names = ["malignant", "benign"]

print("=" * 55)
print("WEEK 4: MACHINE LEARNING MODEL EVALUATION")
print("=" * 55)
print(f"Dataset shape: {X.shape}")
print(f"Missing feature values: {int(X.isna().sum().sum())}")
print(f"Duplicate feature rows: {int(X.duplicated().sum())}")
print("\nClass distribution:")
print(y.map({0: "malignant", 1: "benign"}).value_counts())


# 2. Split the data before fitting preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# 3. Define two models
# Logistic Regression uses scaling because features have different ranges.
logistic_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ]
)

# A limited tree depth helps keep the Decision Tree from becoming too complex.
tree_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
}


# 4. Evaluate models with cross-validation on the training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )
    cv_results.append(
        {
            "model": name,
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
        }
    )
    print(f"\n{name} cross-validation accuracy: {scores.mean():.3f} "
          f"+/- {scores.std():.3f}")

pd.DataFrame(cv_results).to_csv(
    OUTPUT_DIR / "cross_validation_results.csv", index=False
)


# 5. Fit each model and calculate test and training metrics
metrics_rows = []
fitted_models = {}
test_probabilities = {}
test_predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    test_predictions[name] = predictions
    test_probabilities[name] = probabilities

    metrics_rows.append(
        {
            "model": name,
            "train_accuracy": accuracy_score(y_train, train_predictions),
            "test_accuracy": accuracy_score(y_test, predictions),
            "precision_benign": precision_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
            "recall_benign": recall_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
            "f1_benign": f1_score(
                y_test, predictions, pos_label=1, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0,
    )
    filename = name.lower().replace(" ", "_")
    (OUTPUT_DIR / f"{filename}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    print(f"\n{name} classification report:")
    print(report)

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

print("\nTest-set model comparison:")
print(metrics.round(3).to_string(index=False))


# 6. Chart 1: compare evaluation metrics
# Shorter labels make the chart easier to read.
chart_metrics = metrics.set_index("model")[
    ["test_accuracy", "precision_benign", "recall_benign", "f1_benign", "roc_auc"]
].rename(
    columns={
        "test_accuracy": "Accuracy",
        "precision_benign": "Precision",
        "recall_benign": "Recall",
        "f1_benign": "F1-score",
        "roc_auc": "ROC-AUC",
    }
)

ax = chart_metrics.plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))
ax.set_title("Model Performance on the Test Set")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", alpha=0.25)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_metrics_comparison.png", dpi=200)
plt.close()


# 7. Chart 2: confusion matrix for the primary model
primary_name = "Logistic Regression"
primary_model = fitted_models[primary_name]
primary_predictions = test_predictions[primary_name]

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    primary_predictions,
    labels=[0, 1],
    display_labels=class_names,
    cmap="Blues",
    values_format="d",
    ax=ax,
)
ax.set_title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()


# 8. Chart 3: ROC curves for both models
fig, ax = plt.subplots(figsize=(7, 6))
for name, model in fitted_models.items():
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=name,
        ax=ax,
    )

ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
ax.set_title("ROC Curves on the Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
plt.close()


# 9. Save test predictions for later review
prediction_output = pd.DataFrame(
    {
        "actual_class": y_test.map({0: "malignant", 1: "benign"}),
        "logistic_prediction": pd.Series(
            primary_predictions, index=y_test.index
        ).map({0: "malignant", 1: "benign"}),
        "probability_benign": test_probabilities[primary_name],
    },
    index=y_test.index,
)
prediction_output.index.name = "row_index"
prediction_output.to_csv(OUTPUT_DIR / "test_set_predictions.csv")


print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")
print("Review the CSV files and charts before writing the report.")
print("Note: This model is for educational use only, not medical diagnosis.")
