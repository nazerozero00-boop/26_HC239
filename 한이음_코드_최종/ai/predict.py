"""붕괴 예측: 진단 이력의 Bmax 추세를 외삽해 위험 도달 시점 추정.

원리:
  실제 COMSOL 데이터에서 Bmax는 피로가 진행될수록 단조감소한다
  (안정 0.85 → 주의 0.53 → 위험 0.32). Area는 초반에 오르내림이 있어
  추세가 불안정하지만, Bmax는 방향이 일정해 예측에 훨씬 안정적이다.

  최근 이력의 Bmax 기울기를 선형 회귀로 구해, Bmax가 '위험 진입 기준값'에
  도달할 시점을 외삽한다.

한계(발표 방어): 선형 외삽은 '추세가 유지된다'는 가정. COMSOL 물리모델 기반
정밀 예측이 아니라 '경보용 추세 추정'임을 명시할 것.
"""
import numpy as np

# 위험 진입 기준 Bmax (실제 값으로 교체 가능)
#   주의(k=11000) Bmax≈0.53, 위험(k=35000) Bmax≈0.32.
#   위험 진입 경계를 두 값 사이인 0.45로 둔다.
RISK_BMAX = 0.45
MIN_HISTORY = 3          # 예측에 필요한 최소 진단 횟수
RECENT_N = 5             # 추세 계산에 쓸 최근 점 개수 (초반 흔들림 배제)


def predict_time_to_risk(history):
    """진단 이력에서 위험 도달까지 남은 시간을 추정.

    history: [(t, Area, Bmax), ...] 또는 [(t, Area), ...]
             (Bmax 없으면 Area 추세로 대체 — 하위호환)
    반환: dict(months_to_risk, probability, trend)
      months_to_risk: 위험 도달까지 남은 시간(개월). None이면 추정 불가.
      probability: 위험 진입 확률(0~1).
      trend: 'worsening'(악화 중) / 'stable' / 'improving'
    """
    if len(history) < MIN_HISTORY:
        return {"months_to_risk": None, "probability": 0.0, "trend": "stable"}

    # 최근 RECENT_N개만 사용 (초반 흔들림 배제)
    recent = history[-RECENT_N:]
    has_bmax = len(recent[0]) >= 3

    t = np.array([h[0] for h in recent], dtype=float)
    t = t - t[0]
    if t[-1] <= 0:
        t = np.arange(len(recent), dtype=float)

    if has_bmax:
        # Bmax 기반 (권장): 단조감소라 안정적
        y = np.array([h[2] for h in recent], dtype=float)
        slope, _ = np.polyfit(t, y, 1)   # Bmax 기울기 (보통 음수)
        cur = y[-1]

        if cur <= RISK_BMAX:
            # 이미 위험 Bmax 이하 → 위험 진입
            trend = "worsening"
            months = 0.0
            prob = 1.0
        elif slope < -1e-6:
            # Bmax 감소 중(악화) → RISK_BMAX까지 남은 시간
            trend = "worsening"
            months = max((cur - RISK_BMAX) / (-slope), 0.0)
            # 확률: 위험선에 얼마나 가까운지 (0.85→0, 0.45→1)
            span = 0.85 - RISK_BMAX
            prob = float(np.clip((0.85 - cur) / span, 0, 1))
        else:
            # Bmax 증가/유지 → 악화 아님
            trend = "improving" if slope > 1e-6 else "stable"
            months = None
            prob = float(np.clip((0.85 - cur) / (0.85 - RISK_BMAX) * 0.5, 0, 1))
    else:
        # 하위호환: Area 추세 (Bmax 없을 때)
        y = np.array([h[1] for h in recent], dtype=float)
        slope, _ = np.polyfit(t, y, 1)
        trend = "worsening" if slope > 1e-6 else "stable"
        months = None
        prob = 0.0

    return {"months_to_risk": (round(float(months), 1)
                               if months is not None else None),
            "probability": round(prob, 2),
            "trend": trend}


if __name__ == "__main__":
    # Bmax 기반 예시 (피로 진행 = Bmax 감소)
    worsening = [(0, 3000, 0.75), (1, 4000, 0.68), (2, 4500, 0.60),
                 (3, 4900, 0.52)]
    print("악화 중:", predict_time_to_risk(worsening))
    # 이미 위험
    danger = [(0, 5000, 0.50), (1, 4600, 0.44), (2, 3800, 0.38)]
    print("위험 진입:", predict_time_to_risk(danger))
