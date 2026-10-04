# ja_model.py

import numpy as np


MU0 = 4.0 * np.pi * 1e-7

PARAM_NAMES = [
    "Ms",
    "a",
    "k",
    "c",
    "alpha",
]


# ============================================================
# Anhysteretic magnetization
# ============================================================

def anhysteretic_magnetization(
    H_eff,
    Ms,
    a,
):
    """
    Jiles-Atherton anhysteretic magnetization

    Man = Ms * [coth(He/a) - a/He]

    Parameters
    ----------
    H_eff : float
        effective magnetic field [A/m]

    Ms : float
        saturation magnetization [A/m]

    a : float
        anhysteretic shape parameter [A/m]
    """

    x = H_eff / a

    # x = 0 근처에서는
    # coth(x) - 1/x가 수치적으로 불안정하므로
    # Taylor expansion 사용
    if abs(x) < 1e-6:

        langevin = (
            x / 3.0
            - x**3 / 45.0
            + 2.0 * x**5 / 945.0
        )

    else:

        langevin = (
            1.0 / np.tanh(x)
            - 1.0 / x
        )

    return Ms * langevin


# ============================================================
# dMan / dHe
# ============================================================

def dman_dhe(
    H_eff,
    Ms,
    a,
):
    """
    Anhysteretic magnetization derivative:

    dMan / dHe
    """

    x = H_eff / a

    # He -> 0에서 analytic limit 사용
    if abs(x) < 1e-4:

        derivative = (
            Ms / a
            * (
                1.0 / 3.0
                - x**2 / 15.0
                + 2.0 * x**4 / 189.0
            )
        )

        return derivative

    # 큰 x에서 sinh overflow 방지
    if abs(x) > 350:

        csch_squared = 0.0

    else:

        sinh_x = np.sinh(x)

        csch_squared = (
            1.0 / (sinh_x * sinh_x)
        )

    derivative = (
        Ms / a
        * (
            1.0 / x**2
            - csch_squared
        )
    )

    return derivative


# ============================================================
# dM / dH
# ============================================================

def dM_dH(
    H,
    M,
    delta,
    params,
):
    """
    Static isotropic Jiles-Atherton differential equation.

    Parameters
    ----------
    H : float
        current magnetic field

    M : float
        current magnetization

    delta : int
        +1 : H increasing
        -1 : H decreasing

    params : array-like
        [Ms, a, k, c, alpha]

    Returns
    -------
    dM/dH
    """

    Ms, a, k, c, alpha = params

    # Effective magnetic field
    H_eff = (
        H
        + alpha * M
    )

    # Anhysteretic magnetization
    Man = anhysteretic_magnetization(
        H_eff,
        Ms,
        a,
    )

    # dMan / dHe
    dMan_dHe = dman_dhe(
        H_eff,
        Ms,
        a,
    )

    difference = (
        Man - M
    )

    # --------------------------------------------------------
    # delta_M:
    #
    # field 방향과 irreversible magnetization 방향이
    # 서로 반대이면 irreversible term을 막음
    # --------------------------------------------------------

    if delta * difference > 0:

        delta_M = 1.0

    else:

        delta_M = 0.0


    # Irreversible term denominator
    denominator_irrev = (
        delta * k
        - alpha * difference
    )

    # numerical protection
    eps = 1e-12

    if abs(denominator_irrev) < eps:

        denominator_irrev = (
            eps
            if denominator_irrev >= 0
            else -eps
        )


    irreversible_term = (
        delta_M
        * difference
        / denominator_irrev
    )


    # Chain-rule corrected denominator
    denominator_total = (
        1.0
        + c
        - c
        * alpha
        * dMan_dHe
    )

    if abs(denominator_total) < eps:

        denominator_total = (
            eps
            if denominator_total >= 0
            else -eps
        )


    derivative = (
        irreversible_term
        + c * dMan_dHe
    ) / denominator_total

    return derivative


# ============================================================
# J-A forward simulation
# ============================================================

def simulate_ja(
    H,
    params,
    M0=0.0,
):
    """
    Jiles-Atherton forward simulation.

    H의 시간 순서가 매우 중요함.
    즉 H를 sorting 하면 안 됨.

    Parameters
    ----------
    H : array-like
        magnetic field sequence [A/m]

    params : array-like
        [Ms, a, k, c, alpha]

    M0 : float
        initial magnetization [A/m]

    Returns
    -------
    B : ndarray
        magnetic flux density [T]

    M : ndarray
        magnetization [A/m]
    """

    H = np.asarray(
        H,
        dtype=float,
    )

    params = np.asarray(
        params,
        dtype=float,
    )

    if H.ndim != 1:

        raise ValueError(
            "H는 1차원 배열이어야 합니다."
        )

    if len(H) < 2:

        raise ValueError(
            "H 데이터가 너무 짧습니다."
        )

    if len(params) != 5:

        raise ValueError(
            "params = [Ms, a, k, c, alpha] "
            "5개가 필요합니다."
        )


    M = np.zeros_like(
        H,
        dtype=float,
    )

    M[0] = M0


    # ========================================================
    # RK4 integration
    #
    # 큰 H 범위를 복원할 때 한 샘플의 dH가 너무 커지면
    # RK4가 수치적으로 폭주할 수 있다.
    #
    # 따라서 출력 샘플 수는 그대로 유지하되,
    # 내부에서 |dH| <= MAX_DH_STEP가 되도록 substep한다.
    # 실측 fitting 데이터처럼 dH가 작은 경우에는 기존과 동일하다.
    # ========================================================

    MAX_DH_STEP = 1.0

    for i in range(
        1,
        len(H),
    ):

        H0 = H[i - 1]
        H1 = H[i]

        dH_total = (
            H1 - H0
        )

        # 같은 H가 반복된 경우
        if abs(dH_total) < 1e-15:

            M[i] = M[i - 1]

            continue

        n_substeps = max(
            1,
            int(
                np.ceil(
                    abs(dH_total)
                    / MAX_DH_STEP
                )
            ),
        )

        dH = (
            dH_total
            / n_substeps
        )

        M_sub = M[i - 1]
        H_sub = H0

        for _ in range(
            n_substeps
        ):

            H_next = (
                H_sub
                + dH
            )

            delta = (
                1
                if dH > 0
                else -1
            )

            # --------------------------------
            # RK4 substep
            # --------------------------------

            k1 = dM_dH(
                H_sub,
                M_sub,
                delta,
                params,
            )

            k2 = dM_dH(
                H_sub + dH / 2.0,
                M_sub + dH * k1 / 2.0,
                delta,
                params,
            )

            k3 = dM_dH(
                H_sub + dH / 2.0,
                M_sub + dH * k2 / 2.0,
                delta,
                params,
            )

            k4 = dM_dH(
                H_next,
                M_sub + dH * k3,
                delta,
                params,
            )

            M_sub = (
                M_sub
                + dH
                * (
                    k1
                    + 2.0 * k2
                    + 2.0 * k3
                    + k4
                )
                / 6.0
            )

            H_sub = H_next

        M[i] = M_sub


    # Flux density
    B = (
        MU0
        * (
            H + M
        )
    )

    return B, M