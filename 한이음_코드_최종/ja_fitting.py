# ja_fitting.py

import numpy as np
from scipy.optimize import least_squares

from ja_model import simulate_ja


PARAM_NAMES = [
    "Ms",
    "a",
    "k",
    "c",
    "alpha",
]


# ============================================================
# 입력 데이터 검증
# ============================================================

def _validate_bh_data(H, B):

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    if H.ndim != 1:
        raise ValueError("H는 1차원 배열이어야 합니다.")

    if B.ndim != 1:
        raise ValueError("B는 1차원 배열이어야 합니다.")

    if len(H) != len(B):
        raise ValueError(
            f"H/B 길이가 다릅니다. "
            f"H={len(H)}, B={len(B)}"
        )

    if len(H) < 20:
        raise ValueError(
            "J-A fitting을 하기에는 데이터가 너무 적습니다."
        )

    if not np.all(np.isfinite(H)):
        raise ValueError("H에 NaN 또는 inf가 있습니다.")

    if not np.all(np.isfinite(B)):
        raise ValueError("B에 NaN 또는 inf가 있습니다.")

    return H, B


# ============================================================
# Residual
# ============================================================

def _ja_residual(
    params,
    H,
    B_measured,
    start_index,
    B_scale,
    M0,
):
    """
    optimizer가 최소화할 residual

    residual =
        B_JA - B_measured
    """

    try:

        B_pred, _ = simulate_ja(H=H,params=params,M0=M0,)

    except Exception:

        # 비정상 parameter로 J-A 계산이 실패하면
        # 큰 residual을 반환해서 optimizer가 피하도록 한다.
        return np.full(
            len(B_measured) - start_index,
            1e6,
        )

    if not np.all(np.isfinite(B_pred)):

        return np.full(
            len(B_measured) - start_index,
            1e6,
        )

    residual = (B_pred[start_index:]
                - B_measured[start_index:]
    ) / B_scale

    return residual


# ============================================================
# J-A fitting
# ============================================================

def fit_ja(
    H,
    B,
    initial_params,
    lower_bounds,
    upper_bounds,
    start_index=0,
    M0=0.0,
    max_nfev=1000,
    verbose=1,
    loss="soft_l1",
    f_scale=0.05,
):
    """
    Jiles-Atherton parameter fitting

    Parameters
    ----------
    H : array-like
        시간 순서대로 된 H 데이터

    B : array-like
        측정 B 데이터

    initial_params : array-like
        초기값
        [Ms, a, k, c, alpha]

    lower_bounds : array-like
        파라미터 최소값

    upper_bounds : array-like
        파라미터 최대값

    start_index : int
        residual 계산을 시작할 index

        여러 cycle을 돌린 경우
        초기 transient를 버리고
        마지막 안정 cycle만 fitting할 때 사용

    M0 : float
        초기 magnetization

    max_nfev : int
        최대 function evaluation 횟수

    Returns
    -------
    result : dict
    """

    H, B = _validate_bh_data(
        H,
        B,
    )

    initial_params = np.asarray(
        initial_params,
        dtype=float,
    )

    lower_bounds = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper_bounds = np.asarray(
        upper_bounds,
        dtype=float,
    )

    if len(initial_params) != 5:
        raise ValueError(
            "initial_params는 "
            "[Ms, a, k, c, alpha] 5개여야 합니다."
        )

    if len(lower_bounds) != 5:
        raise ValueError(
            "lower_bounds는 5개여야 합니다."
        )

    if len(upper_bounds) != 5:
        raise ValueError(
            "upper_bounds는 5개여야 합니다."
        )

    if np.any(
        initial_params <= lower_bounds
    ):
        raise ValueError(
            "initial_params는 lower_bounds보다 커야 합니다."
        )

    if np.any(
        initial_params >= upper_bounds
    ):
        raise ValueError(
            "initial_params는 upper_bounds보다 작아야 합니다."
        )

    if start_index < 0:
        raise ValueError(
            "start_index는 0 이상이어야 합니다."
        )

    if start_index >= len(H):
        raise ValueError(
            "start_index가 데이터 길이보다 큽니다."
        )


    # --------------------------------------------------------
    # residual scaling
    # --------------------------------------------------------

    B_target = B[start_index:]

    B_scale = np.ptp(B_target)

    if B_scale < 1e-12:
        B_scale = 1.0


    # --------------------------------------------------------
    # parameter scaling
    #
    # Ms ~ 10^5
    # alpha ~ 10^-7
    #
    # 크기 차이가 매우 크기 때문에
    # optimizer scaling이 중요함
    # --------------------------------------------------------

    x_scale = np.maximum(
        np.abs(initial_params),
        1e-12,
    )


    # --------------------------------------------------------
    # optimization
    # --------------------------------------------------------

    optimization = least_squares(
        fun=_ja_residual,
        x0=initial_params,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        args=(
            H,
            B,
            start_index,
            B_scale,
            M0,
        ),
        method="trf",
        x_scale=x_scale,

        # 순간 spike/outlier가 전체 fitting을 끌고 가지 않도록
        # 일반 L2 대신 robust soft-L1 사용
        loss=loss,
        f_scale=f_scale,

        max_nfev=max_nfev,
        verbose=verbose,
    )

    fitted_params = optimization.x


    # --------------------------------------------------------
    # 최적 parameter로 전체 B 재계산
    # --------------------------------------------------------

    B_fitted, M_fitted = simulate_ja(
        H=H,
        params=fitted_params,
        M0=M0,
    )


    # --------------------------------------------------------
    # fitting 구간 RMSE
    # --------------------------------------------------------

    error = (
        B_fitted[start_index:]
        - B[start_index:]
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    B_range = np.ptp(
        B[start_index:]
    )

    if B_range > 1e-12:

        nrmse = (
            rmse
            / B_range
            * 100.0
        )

    else:

        nrmse = np.nan


    param_dict = {

        name: float(value)

        for name, value
        in zip(
            PARAM_NAMES,
            fitted_params,
        )
    }


    return {

        "success":
            bool(optimization.success),

        "message":
            optimization.message,

        "params":
            fitted_params,

        "param_dict":
            param_dict,

        "B_fitted":
            B_fitted,

        "M_fitted":
            M_fitted,

        "rmse":
            float(rmse),

        "nrmse_percent":
            float(nrmse),

        "cost":
            float(optimization.cost),

        "nfev":
            int(optimization.nfev),
    }