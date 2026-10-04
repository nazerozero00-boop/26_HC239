"""측정 CSV 로더 — 실측 33.1 Hz fallback 지원.

지원 형식
1) time,H,B
2) Timestamp,H_Voltage,B_Voltage

Timestamp가 분 단위라 샘플 간격을 계산할 수 없으면 ja_config.DEFAULT_SAMPLING_FREQUENCY
(현재 33.1 Hz)를 사용해 상대 시간축을 생성한다.

주의: H_Voltage/B_Voltage는 절대 H[A/m], B[T]로 환산되기 전 센서 전압일 수 있다.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
import numpy as np
import ja_config as cfg


def _find_col(fieldnames, *candidates):
    low = {f.lower().strip(): f for f in fieldnames if f is not None}
    for c in candidates:
        key = c.lower().strip()
        if key in low:
            return low[key]
    return None


def _parse_time_value(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass

    formats = (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    )
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _usable_time_axis(values, n):
    if len(values) != n or n < 2:
        return None, None
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None, None
    arr = arr - arr[0]
    dt = np.diff(arr)
    if np.any(dt <= 0):
        return None, None
    dt_med = float(np.median(dt))
    if dt_med <= 0:
        return None, None
    return arr, 1.0 / dt_med


def load_measurement(path):
    """CSV 1개를 진단/학습용 측정 dict로 변환한다."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = True
        if not has_header:
            raise ValueError(
                "헤더가 있는 CSV가 필요합니다. 지원: time,H,B 또는 "
                "Timestamp,H_Voltage,B_Voltage"
            )

        reader = csv.DictReader(f)
        fn = reader.fieldnames or []
        tcol = _find_col(fn, "time", "t", "시간", "timestamp", "elapsed_time")
        hcol = _find_col(fn, "h", "hx", "h(a/m)", "h_voltage", "current_voltage")
        bcol = _find_col(fn, "b", "bx", "b(t)", "b_voltage", "hall_voltage")
        kcol = _find_col(fn, "k_fatigue", "k", "kfatigue")
        lcol = _find_col(fn, "true_label", "label", "정답")
        ccol = _find_col(fn, "cycle_id", "cycle")

        if hcol is None or bcol is None:
            raise ValueError(
                f"H/B 열을 찾을 수 없음. 현재 열: {fn}. "
                "H,B 또는 H_Voltage,B_Voltage가 필요합니다."
            )

        rows = list(reader)
        if ccol is not None:
            cyc = {r.get(ccol) for r in rows if r.get(ccol)}
            if len(cyc) > 1:
                raise ValueError(
                    f"이 파일은 측정 {len(cyc)}개가 든 일괄 파일입니다. "
                    "학습/진단 파일 선택에는 측정 1개짜리 CSV를 넣으세요."
                )

        T, H, B, ks, labels = [], [], [], [], []
        for r in rows:
            try:
                hv = float(r[hcol])
                bv = float(r[bcol])
            except (ValueError, TypeError, KeyError):
                continue
            if not (np.isfinite(hv) and np.isfinite(bv)):
                continue
            H.append(hv)
            B.append(bv)

            if tcol and r.get(tcol):
                tv = _parse_time_value(r[tcol])
                T.append(np.nan if tv is None else tv)
            else:
                T.append(np.nan)

            if kcol and r.get(kcol):
                try:
                    ks.append(int(float(r[kcol])))
                except (ValueError, TypeError):
                    pass
            if lcol and r.get(lcol):
                labels.append(str(r[lcol]).strip())

    if len(H) < 20:
        raise ValueError(f"데이터가 너무 짧습니다. 유효 샘플={len(H)}개")

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)
    time, fs = _usable_time_axis(T, len(H))
    used_fallback_fs = False
    if time is None or fs is None:
        fs = float(cfg.DEFAULT_SAMPLING_FREQUENCY)
        time = np.arange(len(H), dtype=float) / fs
        used_fallback_fs = True

    hname = hcol.lower().strip()
    bname = bcol.lower().strip()
    input_kind = (
        "sensor_voltage"
        if ("voltage" in hname or "voltage" in bname)
        else "physical_bh"
    )

    return {
        "time": time,
        "H": H,
        "B": B,
        "fs": float(fs),
        "k": ks[0] if ks else None,
        "true_label": labels[0] if labels else None,
        "input_kind": input_kind,
        "used_fallback_fs": used_fallback_fs,
        "source_columns": {"time": tcol, "H": hcol, "B": bcol},
        "source_path": os.path.abspath(path),
        "source_name": os.path.basename(path),
    }
