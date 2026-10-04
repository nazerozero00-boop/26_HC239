"""SQLite 저장소 — 진단 이력 + 실측 재학습 이력.

DB 파일: ai/diagnosis_training_history.db
- diagnosis_history: 진단 결과 영구 저장
- training_samples: 학습 모드에서 선택한 파일의 특징/라벨 저장
"""
from __future__ import annotations

import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "diagnosis_training_history.db")

_CREATE_DIAGNOSIS = """
CREATE TABLE IF NOT EXISTS diagnosis_history (
    diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    structure_name TEXT NOT NULL DEFAULT '미입력',
    measurement_point TEXT NOT NULL DEFAULT '미입력',
    source_mode TEXT NOT NULL DEFAULT 'unknown',
    Bmax REAL NOT NULL,
    Hmax REAL NOT NULL,
    Area REAL NOT NULL,
    dArea REAL NOT NULL,
    label TEXT NOT NULL,
    label_idx INTEGER NOT NULL
)
"""

_CREATE_TRAINING = """
CREATE TABLE IF NOT EXISTS training_samples (
    training_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    source_file TEXT NOT NULL,
    structure_name TEXT NOT NULL DEFAULT '미입력',
    measurement_point TEXT NOT NULL DEFAULT '미입력',
    input_kind TEXT NOT NULL DEFAULT 'unknown',
    Bmax REAL NOT NULL,
    Hmax REAL NOT NULL,
    Area REAL NOT NULL,
    nrmse REAL,
    label TEXT NOT NULL,
    label_idx INTEGER NOT NULL
)
"""


def _connect(path=DB_PATH):
    return sqlite3.connect(path)


def init_db(path=DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _connect(path) as conn:
        conn.execute(_CREATE_DIAGNOSIS)
        conn.execute(_CREATE_TRAINING)
        conn.commit()
    return path


def insert_diagnosis(bmax, hmax, area, darea, label, label_idx,
                     structure_name="", measurement_point="",
                     source_mode="unknown", timestamp=None, path=DB_PATH):
    init_db(path)
    ts = float(time.time() if timestamp is None else timestamp)
    structure_name = (structure_name or "미입력").strip() or "미입력"
    measurement_point = (measurement_point or "미입력").strip() or "미입력"
    with _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO diagnosis_history
              (timestamp, structure_name, measurement_point, source_mode,
               Bmax, Hmax, Area, dArea, label, label_idx)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, structure_name, measurement_point, str(source_mode),
             float(bmax), float(hmax), float(area), float(darea),
             str(label), int(label_idx)),
        )
        conn.commit()
        return cur.lastrowid


def load_diagnosis_records(limit=None, path=DB_PATH):
    init_db(path)
    sql = """
        SELECT diagnosis_id, timestamp, structure_name, measurement_point,
               source_mode, Bmax, Hmax, Area, dArea, label, label_idx
        FROM diagnosis_history
        ORDER BY diagnosis_id DESC
    """
    params = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def insert_training_sample(source_file, label, label_idx, bmax, hmax, area,
                           nrmse=None, structure_name="", measurement_point="",
                           input_kind="unknown", timestamp=None, path=DB_PATH):
    init_db(path)
    ts = float(time.time() if timestamp is None else timestamp)
    structure_name = (structure_name or "미입력").strip() or "미입력"
    measurement_point = (measurement_point or "미입력").strip() or "미입력"
    with _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO training_samples
              (timestamp, source_file, structure_name, measurement_point,
               input_kind, Bmax, Hmax, Area, nrmse, label, label_idx)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, str(source_file), structure_name, measurement_point,
             str(input_kind), float(bmax), float(hmax), float(area),
             None if nrmse is None else float(nrmse), str(label), int(label_idx)),
        )
        conn.commit()
        return cur.lastrowid


def load_training_samples(limit=None, path=DB_PATH):
    init_db(path)
    sql = """
        SELECT training_id, timestamp, source_file, structure_name,
               measurement_point, input_kind, Bmax, Hmax, Area, nrmse,
               label, label_idx
        FROM training_samples
        ORDER BY training_id DESC
    """
    params = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def clear_training_samples(path=DB_PATH):
    init_db(path)
    with _connect(path) as conn:
        conn.execute("DELETE FROM training_samples")
        conn.commit()


def count_diagnosis(path=DB_PATH):
    init_db(path)
    with _connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM diagnosis_history").fetchone()[0])


def count_training(path=DB_PATH):
    init_db(path)
    with _connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM training_samples").fetchone()[0])
