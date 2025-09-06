
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.utils import load_heart_data, split_features_target, get_feature_groups

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
SYNTH_PATH = PROJECT_ROOT / "data" / "sample_heart_disease.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_groups(X)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def evaluate_classifier(name: str, clf: Pipeline, X_test, y_test) -> Dict[str, Any]:
    y_pred = clf.predict(X_test)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        if hasattr(clf, "decision_function"):
            df = clf.decision_function(X_test)
            df_min, df_max = df.min(), df.max()
            y_proba = (df - df_min) / (df_max - df_min + 1e-9)

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def plot_roc_curves(name_to_proba_truth: List[Tuple[str, np.ndarray, np.ndarray]], out_path: Path):
    plt.figure()
    for name, proba, y_true in name_to_proba_truth:
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, proba)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(train_with_pca: bool = True, random_state: int = 42):
    df = load_heart_data(str(DATA_PATH), allow_synthetic=True, synthetic_path=str(SYNTH_PATH))
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(X)
    pca = PCA(n_components=0.95, random_state=random_state) if train_with_pca else "passthrough"

    estimators = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
    }

    results = []
    proba_blobs = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    for name, est in estimators.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("pca", pca), ("clf", est)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_classifier(name, pipe, X_test, y_test)
        results.append(metrics)

        y_proba = None
        try:
            y_proba = pipe.predict_proba(X_test)[:,1]
        except Exception:
            if hasattr(pipe, "decision_function"):
                dfc = pipe.decision_function(X_test)
                df_min, df_max = dfc.min(), dfc.max()
                y_proba = (dfc - df_min) / (df_max - df_min + 1e-9)
        proba_blobs.append((name, y_proba, y_test.values))

        joblib.dump(pipe, MODELS_DIR / f"{name.lower()}_model.pkl")

    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "evaluation_metrics.txt", "w") as f:
        for m in results:
            f.write(json.dumps(m, indent=2) + "\n")

    plot_roc_curves(proba_blobs, RESULTS_DIR / "roc_curves.png")

    # Hyperparameter tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rf = Pipeline(steps=[("preprocess", preprocessor), ("pca", pca),
                         ("clf", RandomForestClassifier(random_state=random_state))])
    rf_params = {
        "clf__n_estimators": [200, 300, 400, 600],
        "clf__max_depth": [None, 4, 6, 8, 12],
        "clf__min_samples_split": [2, 4, 8],
        "clf__min_samples_leaf": [1, 2, 4]
    }
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=12, cv=cv, scoring="f1", n_jobs=-1, random_state=random_state)
    rf_search.fit(X_train, y_train)

    svm = Pipeline(steps=[("preprocess", preprocessor), ("pca", pca), ("clf", SVC(probability=True, random_state=random_state))])
    svm_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01, 0.1],
        "clf__kernel": ["rbf"]
    }
    svm_search = GridSearchCV(svm, svm_grid, cv=cv, scoring="f1", n_jobs=-1)
    svm_search.fit(X_train, y_train)

    tuned = {"RandomForest_Tuned": rf_search.best_estimator_, "SVM_Tuned": svm_search.best_estimator_}
    tuned_results = []
    for name, model in tuned.items():
        m = evaluate_classifier(name, model, X_test, y_test)
        tuned_results.append(m)
        joblib.dump(model, MODELS_DIR / f"{name.lower()}.pkl")

    all_results = results + tuned_results
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    best = max(tuned_results, key=lambda d: d["f1"])
    final_model = tuned["RandomForest_Tuned"] if best["model"]=="RandomForest_Tuned" else tuned["SVM_Tuned"]
    joblib.dump(final_model, MODELS_DIR / "final_model.pkl")
    print("Training complete. Best tuned model:", best)

if __name__ == "__main__":
    main()
