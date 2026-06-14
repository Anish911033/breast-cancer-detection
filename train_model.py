"""
Train the breast cancer classification model and save the artifacts
(model.pkl, label_encoder.pkl, feature_columns.pkl) that the Streamlit
app (app.py) needs.

Run this ONCE, locally, with your dataset CSV:

    python train_model.py --data breast_cancer_enhanced_dataset.csv

This recreates the same "5 models -> pick the best" approach used in the
notebook (Logistic Regression, Random Forest, Extra Trees,
Gradient Boosting, XGBoost) and saves the best-performing pipeline.
"""

import argparse

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# The 15 feature columns used for prediction (id and diagnosis are excluded)
FEATURE_COLUMNS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "shape_irregularity",
    "border_complexity",
    "tumor_aggressiveness",
    "radius_texture_interaction",
    "radius_concavity_interaction",
    "concavity_density",
    "malignancy_risk_score",
]


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", ExtraTreesClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=4,
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main(data_path: str):
    df = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["diagnosis"])
    X = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()

    print(f"{'Model':<22}{'Accuracy':>10}{'AUC-ROC':>10}{'CV Mean':>10}")
    print("-" * 52)

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(clone(model), X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"model": model, "accuracy": acc, "auc": auc, "cv_mean": cv_scores.mean()}
        print(f"{name:<22}{acc * 100:>9.2f}%{auc:>10.4f}{cv_scores.mean() * 100:>9.2f}%")

    best_name = max(results, key=lambda m: results[m]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}  (test accuracy = {results[best_name]['accuracy'] * 100:.2f}%)")

    joblib.dump(best_model, "model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")
    print("\nSaved: model.pkl, label_encoder.pkl, feature_columns.pkl")
    print("These 3 files must be committed to your repo for the Streamlit app to work.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the breast cancer model")
    parser.add_argument(
        "--data",
        default="breast_cancer_enhanced_dataset.csv",
        help="Path to breast_cancer_enhanced_dataset.csv",
    )
    args = parser.parse_args()
    main(args.data)
