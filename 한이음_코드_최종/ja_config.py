"""J-A/실측 파이프라인 설정 — Arduino UDP 실측 기준."""

# Arduino loop의 delay(10 ms)만 보면 100 Hz지만 UDP 전송까지 포함한 기존 실측은 약 33.1 Hz.
# sample-level timestamp가 없는 과거 CSV에만 fallback으로 사용한다.
DEFAULT_SAMPLING_FREQUENCY = 33.1
ARDUINO_PHASE_STEP_RAD = 0.05  # timeVar += 0.05 rad / measurement loop

# Jiles-Atherton 파라미터 탐색 범위 [Ms, a, k, c, alpha]
JA_LOWER_BOUNDS = [1e4, 1.0, 1.0, 0.01, 1e-6]
JA_UPPER_BOUNDS = [1e6, 5000.0, 1000.0, 0.99, 1e-2]

# H 채널 FFT에서 확인한 노이즈 후보. fs=33.1 Hz의 Nyquist는 16.55 Hz.
# 현재 방침: H만 notch, B는 응답 신호 보존을 위해 RAW 유지.
NOISE_FREQUENCIES = [5.74, 6.77]
NOTCH_Q = 50.0
FILTER_H = True
FILTER_B = False

# J-A 추정 loop 복원
# 현재 프로토타입에서는 복원 범위를 통일하기 위해 10000으로 고정.
H_MAJOR_MAX = 10000.0
MAJOR_POINTS_PER_CYCLE = 2000
MAJOR_CYCLES = 5

# J-A multi-start
N_STARTS = 6
RANDOM_SEED = 42
MAX_NFEV = 1000

# 철근 감지 임계값
REBAR_BMAX_MIN = 0.1
