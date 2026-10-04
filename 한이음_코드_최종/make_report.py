"""active MLP 일괄 테스트 혼동행렬 생성."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from run_test import load_test_csv
from diagnosis_engine import DiagnosisEngine
from ai.mlp_model import load_model

LABELS = ["안정", "주의", "위험"]


def run_diagnoses():
    tests = load_test_csv()
    eng = DiagnosisEngine()
    model = load_model()
    idx_map = {name: i for i, name in enumerate(LABELS)}
    y_true, y_pred = [], []
    for _, true_label, H, B in tests:
        fs = 33.1
        t = np.arange(len(H), dtype=float) / fs
        feat = eng.extract_features(t, H, B, fs=fs, noise_frequencies=[])
        x = np.array([[feat["Bmax"], feat["Hmax"], feat["Area"]]])
        y_true.append(idx_map[true_label])
        y_pred.append(int(model.predict(x)[0]))
    return np.asarray(y_true), np.asarray(y_pred)


def make_confusion_png(y_true, y_pred, out="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    acc = float((y_true == y_pred).mean())
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["Stable", "Caution", "Danger"])
    ax.set_yticklabels(["Stable", "Caution", "Danger"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix / Accuracy {acc*100:.1f}%")
    vmax = max(int(cm.max()), 1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax*0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return cm, acc


def print_metrics(y_true, y_pred):
    prec = precision_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    print(f"전체 정확도: {(y_true==y_pred).mean()*100:.1f}%")
    for i, name in enumerate(LABELS):
        print(f"{name}: 정밀도 {prec[i]*100:.1f}% / 재현율 {rec[i]*100:.1f}%")


if __name__ == "__main__":
    yt, yp = run_diagnoses()
    cm, _ = make_confusion_png(yt, yp)
    print(cm)
    print_metrics(yt, yp)
