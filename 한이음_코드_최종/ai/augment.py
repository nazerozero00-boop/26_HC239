"""학습 데이터 증강: COMSOL 10점 → 물리 노이즈 낀 학습 데이터.

핵심 원리:
  보간만 하면 생성점이 곡선 위에 딱 붙어 AI가 배울 게 없다(= interp 재현).
  물리 노이즈를 더해 곡선 근방에 퍼뜨려야 AI가 "노이즈 속 경향 추출"을
  학습하고, 실측 노이즈에도 강건해진다.

두 단계:
  1) 보간 — k를 조밀하게 만들고, 각 k에서 Area/Bmax/Hmax를 PCHIP 보간.
     PCHIP은 단조 구간에서 오버슈트(없는 봉우리) 없이 이어 물리적으로 안전.
  2) 물리 노이즈 — 세 종류를 근거 있게 주입:
     (a) 센서 정확도 오차: 값에 비례(±%), 가우시안. 홀센서/전류센서 스펙 기반.
     (b) ADC 양자화 오차: 균일분포, 값 크기와 무관. ADC 비트수 기반.
     (c) 온도 드리프트: 한 방향으로 천천히 쏠림(편향). 측정 세션 내 온도 변화.

각 노이즈 세기는 상수로 분리 — 실제 센서 스펙 확정 시 값만 교체.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from ai.comsol_data import as_arrays, label_from_k


# ── 물리 노이즈 파라미터 (실제 센서 스펙 확정 시 교체) ──────────────
SENSOR_ACCURACY = {      # (a) 센서 정확도: 값에 비례하는 표준편차 비율
    "Area": 0.02,        #   면적 ±2%
    "Bmax": 0.015,       #   홀센서 B ±1.5% (데이터시트 정확도 예시)
    "Hmax": 0.02,        #   전류→H ±2%
}
ADC_BITS = 12            # (b) ADC 비트수 → 양자화 스텝
ADC_FULLSCALE = {        #   각 채널의 풀스케일(양자화 계산용, 근사)
    "Area": 6000.0,
    "Bmax": 1.0,
    "Hmax": 17000.0,
}
TEMP_DRIFT = {           # (c) 온도 드리프트: 세션 내 한 방향 최대 쏠림(비율)
    "Area": 0.01,        #   면적 최대 ±1% 편향
    "Bmax": 0.02,        #   홀센서는 온도 민감 → B 최대 ±2%
    "Hmax": 0.01,
}
# ────────────────────────────────────────────────────────────


def _interpolate(k_dense):
    """COMSOL 점을 PCHIP로 보간해 각 k_dense에서 Area/Bmax/Hmax 반환."""
    k, area, bmax, hmax = as_arrays()
    # k는 단조 증가여야 PchipInterpolator가 동작 (COMSOL은 진행순 = k증가순)
    order = np.argsort(k)
    k, area, bmax, hmax = k[order], area[order], bmax[order], hmax[order]
    fa = PchipInterpolator(k, area)
    fb = PchipInterpolator(k, bmax)
    fh = PchipInterpolator(k, hmax)
    return fa(k_dense), fb(k_dense), fh(k_dense)


def _apply_physical_noise(values, name, rng):
    """한 특징 배열에 세 종류 물리 노이즈를 더한다."""
    v = values.astype(float).copy()
    n = len(v)

    # (a) 센서 정확도: 값에 비례하는 가우시안
    sigma = SENSOR_ACCURACY[name] * np.abs(v)
    v = v + rng.normal(0, sigma)

    # (b) ADC 양자화: 균일분포 (±0.5 스텝), 값 크기 무관
    step = ADC_FULLSCALE[name] / (2 ** ADC_BITS)
    v = v + rng.uniform(-0.5 * step, 0.5 * step, size=n)

    # (c) 온도 드리프트: 세션 내 한 방향 선형 쏠림 (샘플마다 다른 방향)
    drift_max = TEMP_DRIFT[name] * np.abs(v).mean()
    direction = rng.choice([-1, 1])
    ramp = np.linspace(0, drift_max * direction, n)
    v = v + ramp

    return v


def generate(per_class=1500, n_dense=6000, seed=0, balance=True):
    """학습 데이터 생성 (클래스 균형 옵션).

    per_class: balance=True일 때 각 단계(안정/주의/위험)당 목표 샘플 수
    n_dense: k 조밀화 개수 (충분히 크게 잡아 각 구간에서 뽑을 여유 확보)
    balance: True면 세 단계를 per_class개씩 균형 맞춤 (불균형 방지)
    반환: X (N,2) = [Bmax, Area], y (N,) = 라벨(0/1/2)

    dArea는 AI 학습 입력에서 제외한다. 변화량은 진단 이력/추세 분석용으로만 사용한다.
    """
    rng = np.random.default_rng(seed)
    k, _, _, _ = as_arrays()

    # k를 조밀하게 (로그 스케일 — k가 50~40000으로 넓어 균등 커버)
    k_dense = np.logspace(np.log10(k.min()), np.log10(k.max()), n_dense)
    area, bmax, hmax = _interpolate(k_dense)

    # 물리 노이즈 주입
    area_n = _apply_physical_noise(area, "Area", rng)
    bmax_n = _apply_physical_noise(bmax, "Bmax", rng)
    # Hmax는 현재 프로토타입에서 J-A 복원 범위로 10000에 고정되므로
    # 현재 분류 입력에서는 제외한다.
    # 실제 절대 H 측정/보정이 가능한 최종 시스템에서는
    # 별도 검증 후 Hmax를 다시 입력 특징으로 추가할 수 있다.
    X_all = np.column_stack([
        bmax_n,
        area_n,
    ])
    y_all = np.array([label_from_k(kk) for kk in k_dense], dtype=int)

    if not balance:
        return X_all, y_all

    # 클래스 균형: 각 단계에서 per_class개씩 (부족하면 복원추출로 채움)
    Xs, ys = [], []
    for cls in (0, 1, 2):
        idx = np.where(y_all == cls)[0]
        if len(idx) == 0:
            continue
        replace = len(idx) < per_class
        pick = rng.choice(idx, size=per_class, replace=replace)
        Xs.append(X_all[pick])
        ys.append(y_all[pick])
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # 섞기
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = generate(per_class=1500, seed=0)
    print("생성된 학습 데이터:", X.shape)
    for i, name in enumerate(["안정", "주의", "위험"]):
        print(f"  {name}: {(y == i).sum()}개")
    print("특징 예시 (첫 3행) [Bmax, Area]:")
    print(X[:3].round(3))
