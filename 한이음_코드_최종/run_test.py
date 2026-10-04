"""일괄 합성 B-H 고리 검증 — DB를 오염시키지 않고 2-feature active MLP 성능 확인."""
from __future__ import annotations

import csv
import os
from collections import OrderedDict
import numpy as np

from diagnosis_engine import DiagnosisEngine
from ai.mlp_model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))


def load_test_csv(path=None):
    if path is None:
        path = os.path.join(HERE, "batch_test_18loops.csv")
    cycles = OrderedDict()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = int(r["cycle_id"])
            if cid not in cycles:
                cycles[cid] = {
                    "k": int(r["k_fatigue"]), "label": r["true_label"],
                    "H": [], "B": []
                }
            cycles[cid]["H"].append(float(r["H"]))
            cycles[cid]["B"].append(float(r["B"]))
    return [(d["k"], d["label"], np.asarray(d["H"]), np.asarray(d["B"]))
            for d in cycles.values()]


def main():
    tests = load_test_csv()
    eng = DiagnosisEngine()
    model = load_model()
    labels = ["안정", "주의", "위험"]
    correct = 0
    print(f"{'k':>6} {'참값':>4} {'예측':>4} {'Bmax':>8} {'Area':>10}")
    print("-" * 42)
    for k, true_label, H, B in tests:
        fs = 33.1
        t = np.arange(len(H), dtype=float) / fs
        feat = eng.extract_features(t, H, B, fs=fs, noise_frequencies=[])
        x = np.array([[feat["Bmax"], feat["Area"]]])
        idx = int(model.predict(x)[0])
        pred = labels[idx]
        correct += int(pred == true_label)
        print(f"{k:>6} {true_label:>4} {pred:>4} {feat['Bmax']:>8.3f} {feat['Area']:>10.1f}")
    print("-" * 42)
    print(f"정확도: {correct}/{len(tests)} = {correct/len(tests)*100:.1f}%")


if __name__ == "__main__":
    main()
