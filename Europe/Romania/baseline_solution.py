"""Leakage-safe model comparison and sealed-set prediction for Romania.

Run ``python baseline_solution.py`` to evaluate candidate classifiers, select
the best cross-validated model, write predictions.csv, and invoke the checker.
Use ``--eda`` to also write a compact exploratory plot to eda.png.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
CHANNELS = [f"c{i:02d}" for i in range(1, 49)]
PREDICTIONS = HERE / "predictions.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    sealed = pd.read_csv(DATA / "sealed.csv")
    expected_train = {"record_id", *CHANNELS, "vrancea"}
    expected_sealed = {"record_id", *CHANNELS}
    if set(train.columns) != expected_train or set(sealed.columns) != expected_sealed:
        raise ValueError("Unexpected train/sealed columns")
    if train.isna().any().any() or sealed.isna().any().any():
        raise ValueError("Missing values must be handled before fitting")
    if train["record_id"].duplicated().any() or sealed["record_id"].duplicated().any():
        raise ValueError("record_id must be unique within each file")
    return train, sealed


def make_models() -> dict[str, object]:
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=10, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        ),
        "SVM_RBF": make_pipeline(
            StandardScaler(), SVC(C=3.0, gamma="scale", probability=True, random_state=42)
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500,
            max_leaf_nodes=63,
            min_samples_leaf=10,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def plot_eda(train: pd.DataFrame) -> None:
    """Save target/channel distributions and channel-correlation heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(data=train, x="vrancea", ax=axes[0])
    axes[0].set_title("Target counts")
    sns.boxplot(data=train.melt(id_vars="vrancea", value_vars=CHANNELS),
                x="vrancea", y="value", ax=axes[1], showfliers=False)
    axes[1].set_title("Channel distributions by class")
    sns.heatmap(train[CHANNELS].corr(), cmap="coolwarm", center=0,
                vmin=-0.2, vmax=0.2, ax=axes[2], cbar=False)
    axes[2].set_title("Channel correlation")
    fig.tight_layout()
    fig.savefig(HERE / "eda.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {HERE / 'eda.png'}")


def report_eda(train: pd.DataFrame, sealed: pd.DataFrame) -> None:
    x = train[CHANNELS]
    target_counts = train["vrancea"].value_counts().sort_index().to_dict()
    zscores = ((x - x.mean()) / x.std()).abs()
    corr = x.corr().abs().to_numpy()
    max_corr = float(corr[np.triu_indices_from(corr, k=1)].max())
    print(f"Rows: train={len(train)} sealed={len(sealed)}; channels={len(CHANNELS)}")
    print(f"Class counts: {target_counts}; event prevalence={train.vrancea.mean():.4f}")
    print(
        "Missing values: 0; duplicate rows: "
        f"train={train.duplicated().sum()}, sealed={sealed.duplicated().sum()}; "
        f"duplicate channel vectors: train={train.duplicated(CHANNELS).sum()}, "
        f"sealed={sealed.duplicated(CHANNELS).sum()}"
    )
    print(f"|z|>4 channel cells: {(zscores > 4).sum().sum()}; max abs inter-channel corr={max_corr:.3f}")
    class_means = train.groupby("vrancea")[CHANNELS].mean()
    shifts = (class_means.loc[1] - class_means.loc[0]).abs().sort_values(ascending=False)
    print(f"Largest class mean shifts: {shifts.head(8).round(3).to_dict()}")
    train_std = x.std().replace(0, np.nan)
    sealed_shift = ((sealed[CHANNELS].mean() - x.mean()) / train_std).abs()
    print(f"Max standardized train-to-sealed mean shift: {sealed_shift.max():.3f}")
    # The vault table/tag/word are deliberately not predictors or label inputs.
    vault = np.load(DATA / "vault.npz")
    print(f"Vault public dimensions: n={int(vault['n'])}, k={int(vault['k'])}; encoded table only")
    need = (int(vault["n"]) + int(vault["k"]) + 1) // 2
    print(f"Evaluator correction threshold: at least {need}/{int(vault['n'])} agreements")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda", action="store_true", help="write EDA plots to eda.png")
    args = parser.parse_args()

    train, sealed = load_data()
    report_eda(train, sealed)
    if args.eda:
        plot_eda(train)

    X = train[CHANNELS].to_numpy(dtype=np.float64)
    y = train["vrancea"].to_numpy(dtype=int)
    X_sealed = sealed[CHANNELS].to_numpy(dtype=np.float64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    models = make_models()
    results: dict[str, dict[str, float]] = {}
    print("\nLeakage-safe 5-fold stratified CV:")
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {metric: float(scores[f"test_{metric}"].mean()) for metric in scoring}
        print(name, " ".join(f"{key}={value:.4f}" for key, value in results[name].items()))

    # Select by accuracy to align with the evaluator, using recall as a
    # secondary tie-breaker because missed events are costly.
    best_name = max(results, key=lambda name: (results[name]["accuracy"], results[name]["recall"]))
    best_model = models[best_name]
    oof_probability = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.arange(0.35, 0.651, 0.01)
    threshold_results = []
    for threshold in thresholds:
        predicted = (oof_probability >= threshold).astype(int)
        accuracy = float(np.mean(predicted == y))
        recall = float(recall_score(y, predicted, zero_division=0))
        threshold_results.append((accuracy, recall, -float(threshold), float(threshold)))
    best_accuracy, best_recall, _, threshold = max(threshold_results)
    oof_prediction = (oof_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, oof_prediction, labels=[0, 1]).ravel()
    print(
        f"\nSelected model: {best_name}; OOF threshold={threshold:.2f}; "
        f"accuracy={best_accuracy:.4f}; recall={best_recall:.4f}; "
        f"precision={precision_score(y, oof_prediction):.4f}; "
        f"F1={f1_score(y, oof_prediction):.4f}; "
        f"confusion_matrix=[[{tn}, {fp}], [{fn}, {tp}]]"
    )

    best_model.fit(X, y)
    probabilities = best_model.predict_proba(X_sealed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    output = pd.DataFrame({"record_id": sealed["record_id"], "vrancea": predictions})
    if len(output) != len(sealed) or output["record_id"].tolist() != sealed["record_id"].tolist():
        raise RuntimeError("Prediction rows must preserve the sealed record order")
    output.to_csv(PREDICTIONS, index=False)
    print(f"Wrote {len(output)} ordered predictions to {PREDICTIONS}")
    print(f"Prediction counts: {output.vrancea.value_counts().sort_index().to_dict()}")

    print("Evaluator:")
    subprocess.run(
        [sys.executable, str(HERE / "evaluation_script.py"), str(PREDICTIONS)],
        check=True,
    )


if __name__ == "__main__":
    main()
