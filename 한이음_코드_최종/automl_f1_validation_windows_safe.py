"""
AutoML-style + F1 검증 통합 스크립트

현재 프로토타입 분류 입력:
    X = [Bmax, Area]

기능:
1) COMSOL 기반 synthetic train/test 데이터 생성
2) 현재 active MLP 성능 평가
3) 여러 분류 모델 자동 비교(GridSearchCV)
4) Macro F1-score 기준 모델 선택
5) Accuracy / Precision / Recall / F1 / Confusion Matrix 출력
6) 결과 CSV 저장
7) best AutoML 모델 별도 저장
"""

from pathlib import Path
import csv
import warnings

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ai.augment import generate
from ai.comsol_data import LABELS
from ai.mlp_model import load_model


TRAIN_PER_CLASS = 1200
TEST_PER_CLASS = 350
TRAIN_SEED = 0
TEST_SEED = 991
CV_SPLITS = 5
RANDOM_STATE = 42
SCORING = "f1_macro"

HERE = Path(__file__).resolve().parent

RESULT_CSV = HERE / "automl_model_comparison.csv"
REPORT_CSV = HERE / "automl_best_classification_report.csv"
CONFUSION_PNG = HERE / "automl_best_confusion_matrix.png"
BEST_MODEL_PATH = HERE / "ai" / "model_automl_best_2feat.pkl"


def evaluate_predictions(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def print_metrics(result):
    print(
        f"{result['model']:<22s} | "
        f"Acc={result['accuracy']*100:6.2f}% | "
        f"Precision={result['precision_macro']*100:6.2f}% | "
        f"Recall={result['recall_macro']*100:6.2f}% | "
        f"F1-macro={result['f1_macro']*100:6.2f}%"
    )


def build_candidates():
    return {
        "Logistic Regression": {
            "estimator": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            "params": {
                "logisticregression__C": [0.1, 1.0, 10.0],
            },
        },

        "SVM-RBF": {
            "estimator": make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
            "params": {
                "svc__C": [1.0, 10.0, 100.0],
                "svc__gamma": ["scale", 0.1, 1.0],
            },
        },

        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [None, 8],
                "min_samples_leaf": [1, 2],
            },
        },

        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.1],
                "max_depth": [2, 3],
            },
        },

        "KNN": {
            "estimator": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(),
            ),
            "params": {
                "kneighborsclassifier__n_neighbors": [3, 5, 9],
                "kneighborsclassifier__weights": ["uniform", "distance"],
            },
        },

        "MLP": {
            "estimator": make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    activation="relu",
                    solver="adam",
                    max_iter=1500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=35,
                    random_state=RANDOM_STATE,
                ),
            ),
            "params": {
                "mlpclassifier__hidden_layer_sizes": [(32,), (64, 32)],
                "mlpclassifier__alpha": [1e-4, 1e-3],
            },
        },
    }


def save_results_csv(results):
    fields = [
        "model",
        "cv_f1_macro",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "best_params",
    ]

    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_classification_report(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for key, value in report.items():
        if isinstance(value, dict):
            rows.append(
                {
                    "class": key,
                    "precision": value.get("precision", ""),
                    "recall": value.get("recall", ""),
                    "f1-score": value.get("f1-score", ""),
                    "support": value.get("support", ""),
                }
            )
        else:
            rows.append(
                {
                    "class": key,
                    "precision": "",
                    "recall": "",
                    "f1-score": value,
                    "support": "",
                }
            )

    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "precision", "recall", "f1-score", "support"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    print("")
    print("=" * 94)
    print("          AutoML-style + F1 VALIDATION")
    print("=" * 94)
    print("입력 feature : [Bmax, Area]")
    print("모델 선택 기준 : Macro F1-score")

    X_train, y_train = generate(
        per_class=TRAIN_PER_CLASS,
        seed=TRAIN_SEED,
    )

    X_test, y_test = generate(
        per_class=TEST_PER_CLASS,
        seed=TEST_SEED,
    )

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    print("")
    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)

    if X_train.ndim != 2 or X_train.shape[1] != 2:
        raise RuntimeError(
            "현재 ai/augment.py가 2-feature [Bmax, Area] 버전이 아닙니다."
        )

    print("")
    print("=" * 94)
    print("1) CURRENT ACTIVE MLP")
    print("=" * 94)

    active_model = load_model(force_reload=True)
    active_pred = active_model.predict(X_test)

    active_result = evaluate_predictions(
        "Current active MLP",
        y_test,
        active_pred,
    )

    active_result["cv_f1_macro"] = ""
    active_result["best_params"] = "current active model"

    print_metrics(active_result)

    print("")
    print(
        classification_report(
            y_test,
            active_pred,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("")
    print("=" * 94)
    print("2) AUTOML MODEL SEARCH")
    print("=" * 94)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    candidates = build_candidates()
    all_results = [active_result]

    best_name = None
    best_model = None
    best_test_f1 = -np.inf
    best_cv_f1 = -np.inf
    best_pred = None
    best_params = None

    for name, config in candidates.items():
        print("")
        print(f"[{name}] searching...")

        search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["params"],
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )

        search.fit(X_train, y_train)
        pred = search.predict(X_test)

        result = evaluate_predictions(
            name,
            y_test,
            pred,
        )

        result["cv_f1_macro"] = float(search.best_score_)
        result["best_params"] = repr(search.best_params_)

        all_results.append(result)

        print(f"Best CV Macro-F1 = {search.best_score_ * 100:.2f}%")
        print("Best params =", search.best_params_)
        print_metrics(result)

        if (
            search.best_score_ > best_cv_f1 + 1e-12
            or (
                abs(search.best_score_ - best_cv_f1) <= 1e-12
                and result["f1_macro"] > best_test_f1
            )
        ):
            best_cv_f1 = float(search.best_score_)
            best_test_f1 = float(result["f1_macro"])
            best_name = name
            best_model = search.best_estimator_
            best_pred = pred
            best_params = search.best_params_

    ranked = sorted(
        all_results,
        key=lambda r: -float(r["f1_macro"]),
    )

    print("")
    print("=" * 94)
    print("3) FINAL COMPARISON")
    print("=" * 94)

    print(
        f"{'Model':<22s} | "
        f"{'Accuracy':>9s} | "
        f"{'Precision':>9s} | "
        f"{'Recall':>9s} | "
        f"{'Macro-F1':>9s}"
    )

    print("-" * 76)

    for r in ranked:
        print(
            f"{r['model']:<22s} | "
            f"{r['accuracy']*100:8.2f}% | "
            f"{r['precision_macro']*100:8.2f}% | "
            f"{r['recall_macro']*100:8.2f}% | "
            f"{r['f1_macro']*100:8.2f}%"
        )

    print("")
    print("=" * 94)
    print("4) BEST AUTOML MODEL")
    print("=" * 94)

    print("Best model :", best_name)
    print(f"Best CV Macro-F1 : {best_cv_f1 * 100:.2f}%")
    print(f"Independent Test Macro-F1 : {best_test_f1 * 100:.2f}%")
    print("Best params :", best_params)

    print("")
    print(
        classification_report(
            y_test,
            best_pred,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, best_pred)
    print(cm)

    save_results_csv(all_results)
    save_classification_report(y_test, best_pred)
    joblib.dump(best_model, BEST_MODEL_PATH)

    try:
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS,
        )

        disp.plot(values_format="d")

        plt.title(
            f"Best AutoML Model - {best_name}"
        )

        plt.tight_layout()
        plt.savefig(CONFUSION_PNG, dpi=200)
        plt.show()

    except Exception as exc:
        print("Confusion matrix image save skipped:", exc)

    print("")
    print("=" * 94)
    print("저장 완료")
    print("=" * 94)
    print("모델 비교 CSV :", RESULT_CSV)
    print("Best classification report :", REPORT_CSV)
    print("Best confusion matrix image :", CONFUSION_PNG)
    print("Best AutoML model :", BEST_MODEL_PATH)

    print("")
    print(
        "※ best AutoML 모델은 별도 저장만 하며 "
        "현재 active MLP를 자동 교체하지 않습니다."
    )


if __name__ == "__main__":
    main()
