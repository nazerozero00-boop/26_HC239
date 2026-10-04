"""건축물별 학습 CSV 관리.

학습 파일은 프로젝트 폴더 아래에 다음처럼 저장한다.
training_data_by_building/
  1번 건축물/
  2번 건축물/
  ...

GUI에서 건축물을 선택하면 해당 폴더의 CSV만 표시된다.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING_ROOT = HERE / "training_data_by_building"
DEFAULT_BUILDINGS = [f"{i}번 건축물" for i in range(1, 11)]


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("건축물 이름이 비어 있습니다.")
    # 경로 구분자/Windows 금지문자를 제거해 폴더 탈출을 막는다.
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    name = name.strip(" .")
    if not name:
        raise ValueError("사용할 수 없는 건축물 이름입니다.")
    return name


def ensure_root() -> Path:
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    return TRAINING_ROOT


def building_dir(building_name: str) -> Path:
    root = ensure_root()
    d = root / _safe_name(building_name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_buildings():
    root = ensure_root()
    found = [p.name for p in root.iterdir() if p.is_dir()]
    ordered = list(DEFAULT_BUILDINGS)
    for name in sorted(found):
        if name not in ordered:
            ordered.append(name)
    return ordered


def list_csv_files(building_name: str):
    d = building_dir(building_name)
    return sorted(
        [p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=lambda s: s.lower(),
    )


def resolve_csv(building_name: str, filename: str) -> str:
    if not filename:
        raise ValueError("CSV 파일이 선택되지 않았습니다.")
    p = building_dir(building_name) / Path(filename).name
    if not p.is_file():
        raise FileNotFoundError(f"등록된 학습 파일을 찾을 수 없습니다: {p.name}")
    return str(p)


def _dedupe_destination(folder: Path, source_name: str) -> Path:
    src = Path(source_name)
    stem = src.stem
    suffix = src.suffix or ".csv"
    dest = folder / f"{stem}{suffix}"
    n = 2
    while dest.exists():
        dest = folder / f"{stem}_{n}{suffix}"
        n += 1
    return dest


def add_csv_files(building_name: str, paths):
    """외부 CSV를 선택한 건축물 폴더에 복사하고 최종 경로 목록을 반환한다."""
    folder = building_dir(building_name)
    added = []
    for raw in paths:
        src = Path(raw)
        if not src.is_file():
            continue
        if src.suffix.lower() != ".csv":
            continue
        dest = _dedupe_destination(folder, src.name)
        shutil.copy2(src, dest)
        added.append(str(dest))
    if not added:
        raise ValueError("추가할 수 있는 CSV 파일이 없습니다.")
    return added
