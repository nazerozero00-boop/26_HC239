import numpy as np
from scipy.optimize import minimize_scalar

from fft_analysis import analyze_bh_signals
from preprocessing import notch_filter
from ja_model import simulate_ja
from ja_fitting import fit_ja


# ============================================================
# 공통 유틸리티
# ============================================================

def _validate_bh(B, H, fs):
    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)

    if B.ndim != 1 or H.ndim != 1:
        raise ValueError("B와 H는 1차원 배열이어야 합니다.")

    if len(B) != len(H):
        raise ValueError(
            f"B/H 길이가 다릅니다. B={len(B)}, H={len(H)}"
        )

    if len(B) < 20:
        raise ValueError("데이터 길이가 너무 짧습니다.")

    if not np.all(np.isfinite(B)):
        raise ValueError("B에 NaN 또는 inf가 포함되어 있습니다.")

    if not np.all(np.isfinite(H)):
        raise ValueError("H에 NaN 또는 inf가 포함되어 있습니다.")

    if fs <= 0:
        raise ValueError("fs는 0보다 커야 합니다.")

    return B, H


def make_triangle_field(
    H_max,
    points_per_cycle=2000,
    cycles=5,
):
    """
    major-loop reconstruction에 사용할
    -Hmax -> +Hmax -> -Hmax triangle H 입력 생성
    """

    if H_max <= 0:
        raise ValueError("H_max는 0보다 커야 합니다.")

    if points_per_cycle < 20:
        raise ValueError(
            "points_per_cycle은 충분히 커야 합니다."
        )

    half = points_per_cycle // 2

    H_up = np.linspace(
        -H_max,
        H_max,
        half,
        endpoint=False,
    )

    # 마지막 점이 다시 -Hmax에 정확히 도달하도록
    # endpoint=True로 두어 cycle을 닫는다.
    H_down = np.linspace(
        H_max,
        -H_max,
        half,
        endpoint=True,
    )

    one_cycle = np.concatenate(
        [H_up, H_down]
    )

    return np.tile(
        one_cycle,
        cycles,
    )


def calculate_loop_area(H, B):
    """
    닫힌 B-H loop 면적 계산

    |∮ B dH|

    B [T], H [A/m] 기준이면
    면적 단위는 J/m^3
    """

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    if len(H) != len(B):
        raise ValueError(
            "H와 B 길이가 다릅니다."
        )

    # 시작/끝점이 완전히 같지 않더라도
    # 적분에서는 폐곡선이 되도록 마지막에 시작점을 붙인다.
    if (
        not np.isclose(H[0], H[-1])
        or not np.isclose(B[0], B[-1])
    ):
        H = np.concatenate(
            [H, H[:1]]
        )
        B = np.concatenate(
            [B, B[:1]]
        )

    return float(
        abs(
            np.trapezoid(
                B,
                H,
            )
        )
    )


def extract_active_cycles(
    H,
    B,
    fs,
    window_sec=0.25,
):
    """
    H 신호의 moving RMS를 이용하여
    실제 자화가 이루어지는 active 구간만 추출한다.
    """

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    window = max(
        5,
        int(round(fs * window_sec)),
    )

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window

    h_rms = np.sqrt(
        np.convolve(
            H ** 2,
            kernel,
            mode="same",
        )
    )

    q25 = np.percentile(h_rms, 25)
    q75 = np.percentile(h_rms, 75)

    threshold = (
        q25
        + 0.25 * (q75 - q25)
    )

    active_mask = h_rms > threshold

    segments = []
    start = None

    for i, active in enumerate(active_mask):

        if active and start is None:
            start = i

        if (
            start is not None
            and (
                not active
                or i == len(active_mask) - 1
            )
        ):
            end = i if active else i - 1

            segments.append(
                (start, end)
            )

            start = None

    if not segments:
        raise RuntimeError(
            "유효한 자화 구간을 찾지 못했습니다."
        )

    lengths = np.array(
        [
            end - start + 1
            for start, end in segments
        ]
    )

    typical_length = np.median(lengths)

    valid_segments = [
        (start, end)
        for start, end in segments
        if (
            end - start + 1
            >= 0.70 * typical_length
        )
    ]

    if not valid_segments:
        raise RuntimeError(
            "완전한 자화 cycle을 찾지 못했습니다."
        )

    H_active = np.concatenate(
        [
            H[start:end + 1]
            for start, end in valid_segments
        ]
    )

    B_active = np.concatenate(
        [
            B[start:end + 1]
            for start, end in valid_segments
        ]
    )

    return (
        H_active,
        B_active,
        valid_segments,
        threshold,
    )


# ============================================================
# Notch bank
# ============================================================

def apply_notch_bank(
    signal,
    fs,
    noise_frequencies,
    q=30.0,
):
    """
    여러 개의 공통 noise frequency에 notch filter를 순차 적용.

    예:
        noise_frequencies = [60, 120, 180]
    """

    filtered = np.asarray(
        signal,
        dtype=float,
    ).copy()

    if noise_frequencies is None:
        return filtered

    nyquist = fs / 2.0

    for freq in noise_frequencies:

        freq = float(freq)

        if freq <= 0:
            raise ValueError(
                f"잘못된 notch frequency: {freq}"
            )

        if freq >= nyquist:
            raise ValueError(
                f"notch frequency {freq} Hz가 "
                f"Nyquist frequency {nyquist} Hz 이상입니다."
            )

        filtered = notch_filter(
            signal=filtered,
            fs=fs,
            notch_freq=freq,
            q=q,
        )

    return filtered


# ============================================================
# Multi-start initial parameter 생성
# ============================================================

def random_initial_params(
    rng,
    lower_bounds,
    upper_bounds,
):
    """
    J-A parameter 순서:
        [Ms, a, k, c, alpha]

    Ms, a, k, alpha:
        log-scale random

    c:
        linear random
    """

    lower = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper = np.asarray(
        upper_bounds,
        dtype=float,
    )

    if len(lower) != 5 or len(upper) != 5:
        raise ValueError(
            "J-A bounds는 5개여야 합니다."
        )

    if np.any(lower >= upper):
        raise ValueError(
            "각 lower bound는 upper bound보다 작아야 합니다."
        )

    params = np.zeros(5)

    # Ms, a, k, alpha
    for idx in [0, 1, 2, 4]:

        if lower[idx] <= 0:
            raise ValueError(
                "Ms, a, k, alpha의 lower bound는 "
                "0보다 커야 log-scale 초기값을 만들 수 있습니다."
            )

        params[idx] = np.exp(
            rng.uniform(
                np.log(lower[idx]),
                np.log(upper[idx]),
            )
        )

    # c
    params[3] = rng.uniform(
        lower[3],
        upper[3],
    )

    return params


# ============================================================
# Multi-start J-A fitting
# ============================================================

def fit_ja_multistart(
    H,
    B,
    lower_bounds,
    upper_bounds,
    n_starts=8,
    fit_start_index=0,
    random_seed=42,
    max_nfev=1500,
):
    """
    여러 초기값으로 J-A fitting을 수행한 뒤
    NRMSE가 가장 작은 결과를 선택한다.
    """

    rng = np.random.default_rng(
        random_seed
    )

    records = []

    for run in range(
        1,
        n_starts + 1,
    ):

        initial_params = random_initial_params(
            rng=rng,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        try:

            result = fit_ja(
                H=H,
                B=B,

                initial_params=initial_params,

                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,

                start_index=fit_start_index,

                M0=0.0,

                max_nfev=max_nfev,

                verbose=0,
            )

            records.append(
                {
                    "run": run,
                    "success": bool(
                        result["success"]
                    ),
                    "nrmse_percent": float(
                        result["nrmse_percent"]
                    ),
                    "rmse": float(
                        result["rmse"]
                    ),
                    "params": np.asarray(
                        result["params"],
                        dtype=float,
                    ),
                    "message": result["message"],
                    "result": result,
                }
            )

        except Exception as exc:

            records.append(
                {
                    "run": run,
                    "success": False,
                    "nrmse_percent": np.inf,
                    "rmse": np.inf,
                    "params": None,
                    "message": str(exc),
                    "result": None,
                }
            )

    valid_records = [
        r
        for r in records
        if r["success"]
        and np.isfinite(
            r["nrmse_percent"]
        )
    ]

    if len(valid_records) == 0:
        raise RuntimeError(
            "모든 J-A multi-start fitting이 실패했습니다."
        )

    best = min(
        valid_records,
        key=lambda r: r[
            "nrmse_percent"
        ],
    )

    return best, records



# ============================================================
# Cycle별 J-A fitting
# ============================================================

def fit_ja_cyclewise(
    H,
    B,
    active_segments,
    lower_bounds,
    upper_bounds,
    n_starts=8,
    random_seed=42,
    max_nfev=1500,
):
    """
    검출된 active cycle을 서로 이어붙이지 않고
    각 cycle을 독립적으로 J-A fitting한다.

    각 cycle 내부에서는 multi-start fitting을 수행하고,
    cycle별 최저 NRMSE 결과를 저장한다.

    최종적으로 cycle별 best 결과 중
    NRMSE가 가장 작은 cycle을 선택한다.
    """

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    cycle_results = []

    for cycle_index, (start, end) in enumerate(
        active_segments,
        start=1,
    ):
        H_cycle = H[start:end + 1]
        B_cycle = B[start:end + 1]

        if len(H_cycle) < 20:
            cycle_results.append(
                {
                    "cycle": cycle_index,
                    "segment": (int(start), int(end)),
                    "points": int(len(H_cycle)),
                    "success": False,
                    "nrmse_percent": np.inf,
                    "rmse": np.inf,
                    "params": None,
                    "best_fit": None,
                    "all_fit_runs": [],
                    "message": "cycle 데이터가 너무 짧습니다.",
                }
            )
            continue

        try:
            cycle_seed = (
                int(random_seed)
                + cycle_index * 1000
            )

            best_fit, all_fit_runs = (
                fit_ja_multistart(
                    H=H_cycle,
                    B=B_cycle,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    n_starts=n_starts,
                    fit_start_index=0,
                    random_seed=cycle_seed,
                    max_nfev=max_nfev,
                )
            )

            cycle_results.append(
                {
                    "cycle": cycle_index,
                    "segment": (int(start), int(end)),
                    "points": int(len(H_cycle)),
                    "success": True,
                    "nrmse_percent": float(
                        best_fit["nrmse_percent"]
                    ),
                    "rmse": float(
                        best_fit["rmse"]
                    ),
                    "params": np.asarray(
                        best_fit["params"],
                        dtype=float,
                    ),
                    "best_fit": best_fit,
                    "all_fit_runs": all_fit_runs,
                    "message": best_fit["message"],
                }
            )

        except Exception as exc:
            cycle_results.append(
                {
                    "cycle": cycle_index,
                    "segment": (int(start), int(end)),
                    "points": int(len(H_cycle)),
                    "success": False,
                    "nrmse_percent": np.inf,
                    "rmse": np.inf,
                    "params": None,
                    "best_fit": None,
                    "all_fit_runs": [],
                    "message": str(exc),
                }
            )

    valid_cycles = [
        r
        for r in cycle_results
        if r["success"]
        and np.isfinite(
            r["nrmse_percent"]
        )
    ]

    if not valid_cycles:
        raise RuntimeError(
            "유효한 cycle별 J-A fitting 결과가 없습니다."
        )

    # --------------------------------------------------------
    # Cross-cycle validation
    #
    # "자기 cycle에만 가장 잘 맞는 parameter" 대신
    # 각 candidate parameter를 모든 active cycle에 다시 적용하고
    # median NRMSE가 가장 낮은 candidate를 최종 선택한다.
    # --------------------------------------------------------

    for candidate in valid_cycles:

        params = candidate["params"]
        validation_nrmse = []

        for start, end in active_segments:

            H_cycle = H[start:end + 1]
            B_cycle = B[start:end + 1]

            B_pred, _ = simulate_ja(
                H=H_cycle,
                params=params,
                M0=0.0,
            )

            error = (
                B_pred
                - B_cycle
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            )

            B_range = float(
                np.ptp(
                    B_cycle
                )
            )

            if B_range > 1e-12:
                nrmse = (
                    rmse
                    / B_range
                    * 100.0
                )
            else:
                nrmse = np.inf

            validation_nrmse.append(
                float(nrmse)
            )

        candidate[
            "validation_nrmse_list"
        ] = validation_nrmse

        candidate[
            "validation_nrmse_median"
        ] = float(
            np.median(
                validation_nrmse
            )
        )

    selected_cycle = min(
        valid_cycles,
        key=lambda r: r[
            "validation_nrmse_median"
        ],
    )

    start, end = selected_cycle["segment"]

    H_selected = H[start:end + 1].copy()
    B_selected = B[start:end + 1].copy()

    best_fit = selected_cycle["best_fit"]

    B_selected_fitted = np.asarray(
        best_fit["result"]["B_fitted"],
        dtype=float,
    )

    return (
        selected_cycle,
        cycle_results,
        H_selected,
        B_selected,
        B_selected_fitted,
    )



# ============================================================
# Cycle별 초기자화 M0 보정
# ============================================================

def _soft_l1_scalar_cost(
    normalized_residual,
    f_scale=0.05,
):
    """
    M0 1개만 찾을 때 사용하는 robust scalar cost.
    ja_fitting.py의 soft-L1 방향과 맞춘다.
    """
    r = np.asarray(
        normalized_residual,
        dtype=float,
    )

    z = (
        r
        / float(f_scale)
    ) ** 2

    rho = 2.0 * (
        np.sqrt(
            1.0 + z
        )
        - 1.0
    )

    return float(
        np.mean(rho)
    )


def optimize_cycle_m0(
    H_cycle,
    B_cycle,
    params,
):
    """
    J-A parameter [Ms,a,k,c,alpha]는 고정하고,
    각 active cycle의 초기자화 M0만 별도로 최적화한다.

    현재 센서 H/B가 절대 단위 보정 전일 수 있으므로
    M0는 물리량 해석용이 아니라
    초기상태 보정용 nuisance parameter로만 사용한다.
    """

    H_cycle = np.asarray(
        H_cycle,
        dtype=float,
    )

    B_cycle = np.asarray(
        B_cycle,
        dtype=float,
    )

    params = np.asarray(
        params,
        dtype=float,
    )

    Ms = abs(
        float(params[0])
    )

    m0_bound = max(
        1.0,
        0.99 * Ms,
    )

    B_scale = max(
        float(
            np.ptp(
                B_cycle
            )
        ),
        1e-12,
    )

    def objective(m0):
        B_pred, _ = simulate_ja(
            H=H_cycle,
            params=params,
            M0=float(m0),
        )

        normalized = (
            B_pred
            - B_cycle
        ) / B_scale

        return _soft_l1_scalar_cost(
            normalized,
            f_scale=0.05,
        )

    opt = minimize_scalar(
        objective,
        bounds=(
            -m0_bound,
            m0_bound,
        ),
        method="bounded",
        options={
            "xatol": 1e-3,
            "maxiter": 120,
        },
    )

    M0 = float(
        opt.x
    )

    B_pred, M_pred = simulate_ja(
        H=H_cycle,
        params=params,
        M0=M0,
    )

    error = (
        B_pred
        - B_cycle
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    B_range = float(
        np.ptp(
            B_cycle
        )
    )

    if B_range > 1e-12:
        nrmse = (
            rmse
            / B_range
            * 100.0
        )
    else:
        nrmse = np.inf

    return {
        "M0": M0,
        "B_fitted": np.asarray(
            B_pred,
            dtype=float,
        ),
        "M_fitted": np.asarray(
            M_pred,
            dtype=float,
        ),
        "rmse": rmse,
        "nrmse_percent": float(
            nrmse
        ),
        "success": bool(
            opt.success
        ),
    }


def evaluate_cycle_m0s(
    H,
    B,
    active_segments,
    params,
):
    """
    선택된 J-A parameter를 고정한 채
    모든 active cycle에 대해 M0만 각각 최적화한다.
    """
    H = np.asarray(
        H,
        dtype=float,
    )

    B = np.asarray(
        B,
        dtype=float,
    )

    results = []

    for cycle_index, (
        start,
        end,
    ) in enumerate(
        active_segments,
        start=1,
    ):

        H_cycle = H[
            start:end + 1
        ]

        B_cycle = B[
            start:end + 1
        ]

        r = optimize_cycle_m0(
            H_cycle=H_cycle,
            B_cycle=B_cycle,
            params=params,
        )

        results.append(
            {
                "cycle": int(
                    cycle_index
                ),
                "segment": (
                    int(start),
                    int(end),
                ),
                "points": int(
                    len(H_cycle)
                ),
                **r,
            }
        )

    valid_nrmse = [
        r["nrmse_percent"]
        for r in results
        if np.isfinite(
            r["nrmse_percent"]
        )
    ]

    valid_rmse = [
        r["rmse"]
        for r in results
        if np.isfinite(
            r["rmse"]
        )
    ]

    if not valid_nrmse:
        raise RuntimeError(
            "cycle별 M0 보정 결과가 유효하지 않습니다."
        )

    return {
        "results": results,
        "median_nrmse_percent": float(
            np.median(
                valid_nrmse
            )
        ),
        "median_rmse": float(
            np.median(
                valid_rmse
            )
        ),
    }


# ============================================================
# 실제 진단 pipeline
# ============================================================

def run_diagnosis(
    B_raw,
    H_raw,
    fs,
    lower_bounds,
    upper_bounds,

    # background FFT 비교 후 결정한 noise 후보
    noise_frequencies=None,

    # notch 설정
    notch_q=30.0,

    # H/B 필터 적용 여부. 현재 설정은 H만 notch, B는 RAW 유지.
    filter_h=False,
    filter_b=False,

    # FFT 확인용
    fft_top_n=10,
    fft_max_freq=None,

    # J-A multi-start
    n_starts=8,
    random_seed=42,
    max_nfev=1500,

    # 기존 호출부와의 호환을 위해 인자는 유지한다.
    # 실제 fitting 구간은 아래 active-cycle 검출로 결정한다.
    fit_start_index=None,
    excitation_freq=None,

    # major-loop reconstruction
    H_major_max=10000.0,
    major_points_per_cycle=2000,
    major_cycles=5,
):
    """
    전체 진단 pipeline.

    흐름
    ----
    raw B/H
        ↓
    FFT 분석
        ↓
    notch bank
        ↓
    active-cycle 추출
        ↓
    multi-start J-A fitting
        ↓
    lowest-NRMSE result 선택
        ↓
    major-loop reconstruction
        ↓
    major-loop area 계산

    중요
    ----
    noise_frequencies는 현재 자동 판정하지 않는다.
    """

    # --------------------------------------------------------
    # 1. 입력 검증
    # --------------------------------------------------------

    B_raw, H_raw = _validate_bh(
        B_raw,
        H_raw,
        fs,
    )

    # --------------------------------------------------------
    # 2. raw FFT 분석
    # --------------------------------------------------------

    fft_before = analyze_bh_signals(
        B=B_raw,
        H=H_raw,
        fs=fs,
        top_n=fft_top_n,
        max_freq=fft_max_freq,
    )

    # --------------------------------------------------------
    # 3. Notch bank preprocessing
    # --------------------------------------------------------

    if filter_b:
        B_filtered = apply_notch_bank(
            signal=B_raw,
            fs=fs,
            noise_frequencies=noise_frequencies,
            q=notch_q,
        )
    else:
        B_filtered = B_raw.copy()

    if filter_h:
        H_filtered = apply_notch_bank(
            signal=H_raw,
            fs=fs,
            noise_frequencies=noise_frequencies,
            q=notch_q,
        )
    else:
        H_filtered = H_raw.copy()

    # --------------------------------------------------------
    # 4. filtered FFT 분석
    # --------------------------------------------------------

    fft_after = analyze_bh_signals(
        B=B_filtered,
        H=H_filtered,
        fs=fs,
        top_n=fft_top_n,
        max_freq=fft_max_freq,
    )

    # --------------------------------------------------------
    # 5. 실제 자화(active) 구간 추출
    # --------------------------------------------------------

    H_active, B_active, active_segments, active_threshold = (
        extract_active_cycles(
            H=H_filtered,
            B=B_filtered,
            fs=fs,
        )
    )

    print("\n========================================")
    print("       ACTIVE CYCLE CHECK")
    print("========================================")
    print("Segments :", active_segments)

    # --------------------------------------------------------
    # 6. Cycle별 독립 J-A fitting
    # --------------------------------------------------------

    (
        selected_cycle_result,
        cycle_fit_results,
        H_fit,
        B_fit,
        B_fit_model,
    ) = fit_ja_cyclewise(
        H=H_filtered,
        B=B_filtered,
        active_segments=active_segments,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_starts=n_starts,
        random_seed=random_seed,
        max_nfev=max_nfev,
    )

    best_fit = selected_cycle_result["best_fit"]
    all_fit_runs = selected_cycle_result["all_fit_runs"]
    fitted_params = selected_cycle_result["params"]

    print("\n========================================")
    print("        CYCLE-WISE J-A CHECK")
    print("========================================")

    for r in cycle_fit_results:
        if r["success"]:
            print(
                f"Cycle {r['cycle']} "
                f"{r['segment']} | "
                f"points={r['points']} | "
                f"own NRMSE={r['nrmse_percent']:.6f}% | "
                f"cross median="
                f"{r.get('validation_nrmse_median', np.nan):.6f}%"
            )
        else:
            print(
                f"Cycle {r['cycle']} "
                f"{r['segment']} | FAILED | "
                f"{r['message']}"
            )

    print("----------------------------------------")
    print(
        "Selected cycle :",
        selected_cycle_result["cycle"],
    )
    print(
        "Selected segment :",
        selected_cycle_result["segment"],
    )
    print(
        "Best own NRMSE :",
        selected_cycle_result["nrmse_percent"],
    )
    print(
        "Cross-cycle median NRMSE :",
        selected_cycle_result[
            "validation_nrmse_median"
        ],
    )
    print(
        "H fit range :",
        np.min(H_fit),
        "~",
        np.max(H_fit),
    )
    print(
        "B fit range :",
        np.min(B_fit),
        "~",
        np.max(B_fit),
    )

    print("\nFitted parameters")
    print("Ms    :", fitted_params[0])
    print("a     :", fitted_params[1])
    print("k     :", fitted_params[2])
    print("c     :", fitted_params[3])
    print("alpha :", fitted_params[4])

    # --------------------------------------------------------
    # 6-1. 선택된 J-A parameter는 유지하고
    #      cycle별 초기자화 M0만 보정
    # --------------------------------------------------------

    m0_eval = evaluate_cycle_m0s(
        H=H_filtered,
        B=B_filtered,
        active_segments=active_segments,
        params=fitted_params,
    )

    m0_cycle_results = (
        m0_eval["results"]
    )

    m0_median_nrmse = float(
        m0_eval[
            "median_nrmse_percent"
        ]
    )

    m0_median_rmse = float(
        m0_eval[
            "median_rmse"
        ]
    )

    print("")
    print("========================================")
    print("        CYCLE M0 CORRECTION")
    print("========================================")

    for r in m0_cycle_results:
        print(
            f"Cycle {r['cycle']} "
            f"{r['segment']} | "
            f"M0={r['M0']:.6f} | "
            f"NRMSE={r['nrmse_percent']:.6f}%"
        )

    print(
        "M0-corrected median NRMSE :",
        m0_median_nrmse,
    )

    selected_cycle_idx = int(
        selected_cycle_result["cycle"]
    ) - 1

    selected_m0_result = (
        m0_cycle_results[
            selected_cycle_idx
        ]
    )

    # GUI/디버깅용 fitting curve도
    # M0 보정된 선택 cycle 결과로 교체
    B_fit_model = np.asarray(
        selected_m0_result[
            "B_fitted"
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # 7. Major-loop reconstruction
    # --------------------------------------------------------

    H_major_all = make_triangle_field(
        H_max=H_major_max,
        points_per_cycle=major_points_per_cycle,
        cycles=major_cycles,
    )

    B_major_all, M_major_all = simulate_ja(
        H=H_major_all,
        params=fitted_params,
        M0=0.0,
    )

    # 초기상태 영향을 줄이기 위해
    # 마지막 cycle만 최종 major loop로 사용
    H_major = H_major_all[
        -major_points_per_cycle:
    ]

    B_major = B_major_all[
        -major_points_per_cycle:
    ]

    M_major = M_major_all[
        -major_points_per_cycle:
    ]

    # --------------------------------------------------------
    # 8. Major-loop area
    # --------------------------------------------------------

    major_area = calculate_loop_area(
        H_major,
        B_major,
    )

    # --------------------------------------------------------
    # 9. 결과 반환
    # --------------------------------------------------------

    return {
        # 원본 / 전처리 데이터
        "B_raw": B_raw,
        "H_raw": H_raw,
        "B_filtered": B_filtered,
        "H_filtered": H_filtered,

        # active-cycle / cycle별 fitting 데이터
        "H_fit": H_fit,
        "B_fit": B_fit,
        "B_fit_model": B_fit_model,
        "fit_loop": (
            H_fit,
            B_fit_model,
        ),
        "active_segments": active_segments,
        "active_threshold": float(active_threshold),
        "fit_points": int(len(H_fit)),
        "selected_cycle": int(
            selected_cycle_result["cycle"]
        ),
        "selected_segment": tuple(
            selected_cycle_result["segment"]
        ),
        "cycle_fit_results": cycle_fit_results,
        "cross_cycle_nrmse_percent": float(
            m0_median_nrmse
        ),
        "zero_m0_cross_cycle_nrmse_percent": float(
            selected_cycle_result[
                "validation_nrmse_median"
            ]
        ),
        "m0_cycle_results": m0_cycle_results,
        "selected_cycle_m0": float(
            selected_m0_result["M0"]
        ),

        # FFT
        "fft_before": fft_before,
        "fft_after": fft_after,
        "noise_frequencies": (
            []
            if noise_frequencies is None
            else list(noise_frequencies)
        ),

        # fitting
        "fit_start_index": 0,
        "selected_run": int(
            best_fit["run"]
        ),
        "fit_success": bool(
            selected_cycle_result["success"]
        ),
        # 최종 GUI 표시용 NRMSE는
        # cycle별 M0 보정 후 대표값(median)을 사용.
        "fit_nrmse_percent": float(
            m0_median_nrmse
        ),
        "fit_rmse": float(
            m0_median_rmse
        ),
        "zero_m0_selected_cycle_nrmse_percent": float(
            selected_cycle_result["nrmse_percent"]
        ),
        "fitted_params": fitted_params,
        "all_fit_runs": all_fit_runs,

        # major loop
        "H_major": H_major,
        "B_major": B_major,
        "M_major": M_major,
        "major_area": float(
            major_area
        ),
    }


# ============================================================
# 콘솔 확인용
# ============================================================

def print_diagnosis_summary(
    result,
):
    """
    GUI 연결 전 콘솔에서 핵심 결과만 확인.
    """

    print("")
    print("========================================")
    print("           DIAGNOSIS SUMMARY")
    print("========================================")

    print(
        f"Noise frequencies : "
        f"{result['noise_frequencies']}"
    )

    print(
        f"Active segments   : "
        f"{result['active_segments']}"
    )

    print(
        f"Fit points        : "
        f"{result['fit_points']}"
    )

    print(
        f"Selected cycle    : "
        f"{result['selected_cycle']}"
    )

    print(
        f"Selected segment  : "
        f"{result['selected_segment']}"
    )

    print(
        f"Selected run      : "
        f"{result['selected_run']}"
    )

    print(
        f"Fit NRMSE (M0)    : "
        f"{result['fit_nrmse_percent']:.6f}%"
    )

    print(
        f"Selected M0       : "
        f"{result.get('selected_cycle_m0', 0.0):.6f}"
    )

    print(
        f"Major-loop area   : "
        f"{result['major_area']:.8f} J/m^3"
    )

    print("")
    print("Fitted J-A parameters:")

    names = [
        "Ms",
        "a",
        "k",
        "c",
        "alpha",
    ]

    for name, value in zip(
        names,
        result["fitted_params"],
    ):
        print(
            f"{name:8s} = "
            f"{value:.8g}"
        )
