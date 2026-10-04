"""안 배운 k값의 (H,B) 홀센서 좌표 테스트 데이터 생성 → CSV.

학습(AI)에 쓰지 않은 독립 테스트용. 로직이 안정→주의→위험을 제대로
구분하는지, 실제 데이터가 오기 전에 검증하기 위한 샘플을 만든다.

방법 (합성 고리):
  COMSOL 원본 점은 노이즈가 심한 '띠'라 그대로 이으면 지그재그가 된다
  (MATLAB도 스플라인+SGolay로 다듬어야 매끈한 고리가 됨). 그래서 여기선
  COMSOL의 실제 특성값(면적/Bmax/Hmax)을 PCHIP로 보간해 '안 배운 k'의
  목표값을 구하고, 그 면적·Bmax·Hmax를 정확히 재현하는 매끈한 히스테리시스
  고리를 합성한다. → 실제 홀센서가 줄 깨끗한 시간순 (H,B)와 같은 형태.

  이렇게 하면 k-Area 관계(COMSOL 그래프)와 Bmax 추세가 그대로 유지되므로,
  진단 로직 검증에 타당하다.
"""
import csv
import numpy as np
import sys, os
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ai"))
from ai.comsol_data import COMSOL_POINTS
from features import loop_area


def _interp_targets():
    """COMSOL 점을 PCHIP 보간해 (k→Area, k→Bmax, k→Hmax) 함수 반환."""
    arr = np.array(COMSOL_POINTS, dtype=float)
    k, area, bmax, hmax = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    order = np.argsort(k)
    k, area, bmax, hmax = k[order], area[order], bmax[order], hmax[order]
    return (PchipInterpolator(k, area),
            PchipInterpolator(k, bmax),
            PchipInterpolator(k, hmax),
            (k.min(), k.max()))


def synth_loop(area_target, bmax, hmax, n=200):
    """면적·Bmax·Hmax를 재현하는 매끈한 히스테리시스 고리 (H,B) 생성.

    자화곡선(tanh)으로 B를 만들고, 고리 벌어짐은 H방향(위상차)으로 준다.
    - B는 자화곡선만 → 최대가 정확히 ±bmax (면적 조정이 Bmax를 안 건드림)
    - H는 ±hmax 왕복 + 벌어짐. 벌어짐 스케일로 면적을 정확히 맞춘다.

    (이전 버전은 면적을 B방향 sin으로 만들어, 면적을 키우면 Bmax가 함께
    부풀어 위험 구간 진단이 어긋났다. H방향으로 벌리면 그 충돌이 없다.)
    """
    t = np.linspace(0, 2 * np.pi, n)
    # B: 자화곡선만 (최대 정확히 ±bmax)
    B = bmax * np.tanh(1.5 * np.cos(t)) / np.tanh(1.5)
    # H: 기본 왕복 + 고리 벌어짐(90도 위상차) — 면적은 H폭이 만듦
    H_base = hmax * np.cos(t)
    H_lag = np.sin(t)
    pts0 = np.column_stack([H_base + hmax * 0.3 * H_lag, B])
    a0 = loop_area(pts0)
    scale = area_target / a0 if a0 > 0 else 1.0
    H = H_base + hmax * 0.3 * H_lag * scale
    return H, B


def make_loop_at_k(k_target):
    """안 배운 k_target의 매끈한 고리 (H,B)를 합성해 반환."""
    fa, fb, fh, (kmin, kmax) = _interp_targets()
    kc = min(max(k_target, kmin), kmax)   # 보간 범위로 제한
    area = float(fa(kc))
    bmax = float(fb(kc))
    hmax = float(fh(kc))
    return synth_loop(area, bmax, hmax)


def label_for_k(k):
    if k < 10000:
        return "안정"
    elif k < 15000:
        return "주의"
    else:
        return "위험"


def save_loop_csv(k, out_path=None, with_answer=True, noise=0.0, seed=0):
    """단일 k의 고리를 CSV 한 파일로 저장 (파일 선택 모드용).

    with_answer=True면 k_fatigue, true_label 컬럼 포함(채점용).
    False면 H,B만 (실제 홀센서 로그 흉내).
    noise>0이면 약간의 측정 노이즈 추가.
    """
    H, B = make_loop_at_k(k)
    if noise > 0:
        rng = np.random.default_rng(seed)
        H = H + rng.normal(0, noise * np.abs(H).max(), len(H))
        B = B + rng.normal(0, noise * np.abs(B).max(), len(B))
    if out_path is None:
        out_path = f"sample_loop_k{k}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if with_answer:
            w.writerow(["H", "B", "k_fatigue", "true_label"])
            lab = label_for_k(k)
            for h, b in zip(H, B):
                w.writerow([f"{h:.4f}", f"{b:.6f}", k, lab])
        else:
            w.writerow(["H", "B"])
            for h, b in zip(H, B):
                w.writerow([f"{h:.4f}", f"{b:.6f}"])
    return out_path


def generate_batch_csv(k_values, out_path="batch_test_18loops.csv", noise=0.0, seed=0):
    """여러 k의 고리를 하나의 CSV로 (run_test.py 일괄 채점용).

    컬럼: cycle_id, k_fatigue, true_label, point_idx, H, B
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cyc, k in enumerate(k_values):
        H, B = make_loop_at_k(k)
        if noise > 0:
            H = H + rng.normal(0, noise * np.abs(H).max(), len(H))
            B = B + rng.normal(0, noise * np.abs(B).max(), len(B))
        lab = label_for_k(k)
        for i, (h, b) in enumerate(zip(H, B)):
            rows.append([cyc, k, lab, i, f"{h:.4f}", f"{b:.6f}"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle_id", "k_fatigue", "true_label", "point_idx", "H", "B"])
        w.writerows(rows)
    return out_path, len(k_values)


if __name__ == "__main__":
    # 안 배운 k값들 (원본 15점 사이사이)
    k_test = [700, 1300, 1800, 2800, 3200, 4000, 5000, 5500, 7000, 9000,
              10000, 12000, 13000, 14000, 16000, 20000, 25000, 30000]
    # 1) 일괄 채점용 한 파일
    path, n = generate_batch_csv(k_test, noise=0.005, seed=1)
    print(f"일괄 테스트: {path} ({n}개 고리)")
    # 2) 파일 선택 모드용 개별 샘플 (정답 포함 5개 + 정답없음 1개)
    for k in [5000, 9000, 12000, 18000, 28000]:
        p = save_loop_csv(k, noise=0.005, seed=k)
        print(f"  샘플: {p}")
    save_loop_csv(15000, out_path="sample_loop_HBonly.csv",
                  with_answer=False, noise=0.005, seed=99)
    print("  샘플: sample_loop_HBonly.csv (정답 없음)")
