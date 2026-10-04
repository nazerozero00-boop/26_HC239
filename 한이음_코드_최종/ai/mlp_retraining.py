"""COMSOL 사전학습 MLP + 실측 파일 기반 재학습.

기본 모델은 COMSOL 기반 증강 데이터로 학습한다.
학습 모드에서 사용자가 라벨을 지정한 실측 파일은 J-A 특징(Bmax/Area)으로
SQLite에 저장되고, 이 특징들을 COMSOL 학습 데이터와 함께 사용해 active MLP를 재학습한다.
"""
from __future__ import annotations

import os
import shutil
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ai.augment import generate

HERE = os.path.dirname(__file__)
BASE_MODEL_PATH = os.path.join(HERE, "model_base_comsol_mlp_2feat.pkl")
ACTIVE_MODEL_PATH = os.path.join(HERE, "model_active_mlp_2feat.pkl")

BASE_PER_CLASS = 1200
REAL_AUGMENT_PER_SAMPLE = 120


def build_mlp(random_state=0):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=35,
            random_state=random_state,
        ),
    )


def train_base_model(save=True):
    X, y = generate(per_class=BASE_PER_CLASS, seed=0)
    model = build_mlp(random_state=0)
    model.fit(X, y)

    X_test, y_test = generate(per_class=350, seed=991)
    acc = float(accuracy_score(y_test, model.predict(X_test)))

    if save:
        joblib.dump(model, BASE_MODEL_PATH)
        joblib.dump(model, ACTIVE_MODEL_PATH)
    return model, {"comsol_validation_accuracy": acc, "base_samples": int(len(y))}


def ensure_models():
    if not os.path.exists(BASE_MODEL_PATH):
        train_base_model(save=True)
    if not os.path.exists(ACTIVE_MODEL_PATH):
        shutil.copy2(BASE_MODEL_PATH, ACTIVE_MODEL_PATH)
    return BASE_MODEL_PATH, ACTIVE_MODEL_PATH


def _augment_real_sample(feature, label_idx, rng, copies=REAL_AUGMENT_PER_SAMPLE):
    """실측 특징 1건을 작은 상대 잡음으로 복제해 재학습 영향력을 확보한다."""
    x = np.asarray(feature, dtype=float).reshape(1, 2)
    scale = np.maximum(
        np.abs(x),
        np.array([[1e-6, 1e-6]]),
    )
    # Bmax 1.5%, Area 2% 수준의 작은 변동만 부여
    rel = np.array([[0.015, 0.020]])
    noise = (
        rng.normal(
            0.0,
            1.0,
            size=(int(copies), 2),
        )
        * scale
        * rel
    )
    X = np.repeat(x, int(copies), axis=0) + noise
    y = np.full(int(copies), int(label_idx), dtype=int)
    return X, y


def retrain_active_model(training_records):
    """COMSOL 기본 데이터 + DB에 누적된 실측 라벨 특징으로 active MLP 재학습."""
    ensure_models()
    X_base, y_base = generate(per_class=BASE_PER_CLASS, seed=0)
    X_parts = [X_base]
    y_parts = [y_base]
    rng = np.random.default_rng(20260829)

    valid_real = 0
    for r in training_records:
        try:
            feature = [float(r["Bmax"]), float(r["Area"])]
            label_idx = int(r["label_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if label_idx not in (0, 1, 2) or not np.all(np.isfinite(feature)):
            continue
        Xr, yr = _augment_real_sample(feature, label_idx, rng)
        X_parts.append(Xr)
        y_parts.append(yr)
        valid_real += 1

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    model = build_mlp(random_state=0)
    model.fit(X, y)
    joblib.dump(model, ACTIVE_MODEL_PATH)

    X_test, y_test = generate(per_class=350, seed=991)
    acc = float(accuracy_score(y_test, model.predict(X_test)))
    return {
        "active_model_path": ACTIVE_MODEL_PATH,
        "comsol_validation_accuracy": acc,
        "base_samples": int(len(y_base)),
        "real_files": int(valid_real),
        "training_rows_total": int(len(y)),
    }


def restore_base_model():
    ensure_models()
    shutil.copy2(BASE_MODEL_PATH, ACTIVE_MODEL_PATH)
    return ACTIVE_MODEL_PATH


if __name__ == "__main__":
    model, report = train_base_model(save=True)
    print("COMSOL MLP 기본 모델 생성 완료")
    print(report)
    print("base:", BASE_MODEL_PATH)
    print("active:", ACTIVE_MODEL_PATH)
