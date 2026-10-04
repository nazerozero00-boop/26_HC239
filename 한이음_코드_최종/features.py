"""면적 + 특징점: 전체 히스테리시스 고리에서 AI 입력을 뽑는다.

(이전의 area.py를 여기 합침 — 면적 계산과 특징 추출이 항상 같이 쓰이므로)

  - loop_area(): 전체 4사분면 고리 내부 면적 (신발끈). 닫힌 고리라 축으로
    닫을 필요 없고, 신발끈이 좌표 부호(음수 포함)를 자동 처리한다.
  - extract_features(): AI 입력용 (Bmax, Hmax, Area) 반환.

ΔArea(면적 변화량)는 이전 사이클과 비교해야 나오므로 fatigue.py에서 계산.
"""
import numpy as np


def shoelace(poly):
    """(N,2) 폐다각형 꼭짓점의 면적 (절댓값)."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def loop_area(dense_points):
    """전체 히스테리시스 고리 점 (M,2)가 감싸는 내부 면적."""
    pts = np.asarray(dense_points, dtype=float)
    if len(pts) < 3:
        return 0.0
    return shoelace(pts)


def extract_features(dense_points):
    """고리 점 (M,2)에서 (Bmax, Hmax, Area)를 반환.

    반환: dict(Bmax, Hmax, Area). H=pts[:,0], B=pts[:,1].
    """
    pts = np.asarray(dense_points, dtype=float)
    if len(pts) < 3:
        return {"Bmax": 0.0, "Hmax": 0.0, "Area": 0.0}
    Hmax = float(np.max(pts[:, 0]))
    Bmax = float(np.max(pts[:, 1]))
    Area = float(loop_area(pts))
    return {"Bmax": Bmax, "Hmax": Hmax, "Area": Area}
