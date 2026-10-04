"""현재 실행 세션의 진단 이력 CSV.

SQLite는 영구 저장용, 이 CSV는 GUI 추이/예측용이다.
"""
from __future__ import annotations

import csv
import os
import time
from ai import sqlite_store

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "diagnosis_session_history.csv")
_HEADER = [
    "timestamp", "structure_name", "measurement_point", "source_mode",
    "Bmax", "Hmax", "Area", "dArea", "label", "label_idx"
]


def append_record(bmax, hmax, area, darea, label, label_idx,
                  structure_name="", measurement_point="", source_mode="unknown",
                  path=HISTORY_PATH, save_to_db=True):
    timestamp = time.time()
    structure_name = (structure_name or "미입력").strip() or "미입력"
    measurement_point = (measurement_point or "미입력").strip() or "미입력"
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(_HEADER)
        w.writerow([
            f"{timestamp:.3f}", structure_name, measurement_point, source_mode,
            f"{bmax:.6f}", f"{hmax:.6f}", f"{area:.6f}", f"{darea:.6f}",
            label, int(label_idx)
        ])

    if save_to_db:
        sqlite_store.insert_diagnosis(
            bmax, hmax, area, darea, label, label_idx,
            structure_name=structure_name,
            measurement_point=measurement_point,
            source_mode=source_mode,
            timestamp=timestamp,
        )


def load_history(path=HISTORY_PATH):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append({
                    "timestamp": float(row["timestamp"]),
                    "structure_name": row.get("structure_name", "미입력"),
                    "measurement_point": row.get("measurement_point", "미입력"),
                    "source_mode": row.get("source_mode", "unknown"),
                    "Bmax": float(row["Bmax"]),
                    "Hmax": float(row["Hmax"]),
                    "Area": float(row["Area"]),
                    "dArea": float(row["dArea"]),
                    "label": row["label"],
                    "label_idx": int(row["label_idx"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
    return out


def area_series(path=HISTORY_PATH):
    return [(r["timestamp"], r["Area"], r["Bmax"]) for r in load_history(path)]


def clear_history(path=HISTORY_PATH):
    if os.path.exists(path):
        os.remove(path)
