"""COMSOL export(txt) 파싱 → k별 (Area, Bmax, Hmax) 계산.

원본 COMSOL 결과 파일을 다시 파싱해 ai/comsol_data.py의 COMSOL_POINTS를
재생성할 때 사용. 새 COMSOL 데이터가 나오면 RAW_FILES 경로만 바꿔 실행.

각 k_fatigue마다 (Hx, Bx) 시계열이 히스테리시스 고리 하나를 그린다.
사용:  python parse_comsol.py   → k별 (Area,Bmax,Hmax) 출력
"""
import re
import numpy as np
from collections import OrderedDict
from features import loop_area

# COMSOL 원본 파일 경로 (실제 데이터가 있는 위치로 수정)
RAW_FILES = [
    "txt6.txt",
    "txt7.txt",
]


def parse_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    header_line = data_line = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("% x") and "mf.Bx" in s:
            header_line = s
        elif s and not s.startswith("%"):
            data_line = s
    col_specs = []
    for m in re.finditer(
            r'mf\.(Bx|Hx)\s*\([^)]*\)\s*@\s*t=[\d.]+,\s*k_fatigue=(\d+)',
            header_line):
        col_specs.append((m.group(1), int(m.group(2))))
    vals = [float(x) for x in data_line.split()]
    data = vals[2:]
    kdata = OrderedDict()
    for (typ, k), v in zip(col_specs, data):
        kdata.setdefault(k, {"Bx": [], "Hx": []})[typ].append(v)
    return kdata


def parse_all():
    merged = OrderedDict()
    for path in RAW_FILES:
        for k, d in parse_file(path).items():
            merged[k] = d
    return OrderedDict(sorted(merged.items()))


def compute():
    results = []
    for k, d in parse_all().items():
        B = np.array(d["Bx"]); H = np.array(d["Hx"])
        n = min(len(B), len(H)); B, H = B[:n], H[:n]
        pts = np.column_stack([H, B])
        results.append((k, loop_area(pts), float(np.max(B)),
                        float(np.max(H)), n))
    return results


if __name__ == "__main__":
    print(f"{'k':>7} {'Area':>12} {'Bmax':>8} {'Hmax':>10}")
    for k, area, bmax, hmax, n in compute():
        print(f"{k:>7} {area:>12.2f} {bmax:>8.4f} {hmax:>10.1f}")
