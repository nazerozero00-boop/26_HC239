from pathlib import Path
import csv
import numpy as np

import ja_config as cfg

from measurement_csv_33hz import load_measurement
from pipeline import (
    run_diagnosis,
    make_triangle_field,
    calculate_loop_area,
)
from ja_model import simulate_ja


# ============================================================
# 설정
# ============================================================

# None이면 실행할 때 파일 선택창이 뜬다.
# 특정 CSV를 고정해서 쓰고 싶으면 경로를 직접 지정하면 된다.
# 예:
# CSV_PATH = Path(r"C:\Users\김나영\Desktop\bh_raw_data.csv")
CSV_PATH = None

# Hmax sweep 후보
H_TEST_LIST = [
    0.18,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10000.0,
    14000.0,
]

# 그래프로 겹쳐 볼 대표 Hmax
PLOT_H_LIST = [
    0.18,
    10.0,
    100.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    14000.0,
]


# ============================================================
# CSV 선택
# ============================================================

def find_csv():
    """
    CSV_PATH가 지정되어 있으면 그 파일을 사용하고,
    None이면 Windows 파일 선택창을 띄운다.
    """

    if CSV_PATH is not None:
        path = Path(CSV_PATH)

        if not path.exists():
            raise FileNotFoundError(
                f"지정한 CSV가 없습니다: {path}"
            )

        return path

    # 파일 선택창
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title="Hmax sweep에 사용할 RAW CSV 선택",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not selected:
        raise RuntimeError(
            "CSV 선택이 취소되었습니다."
        )

    return Path(selected)


# ============================================================
# fitting은 딱 한 번만 수행
# ============================================================

def fit_once(csv_path):
    d = load_measurement(csv_path)

    H = np.asarray(
        d["H"],
        dtype=float,
    )

    B = np.asarray(
        d["B"],
        dtype=float,
    )

    fs = float(
        d["fs"]
    )

    excitation_freq = (
        fs
        * float(cfg.ARDUINO_PHASE_STEP_RAD)
        / (2.0 * np.pi)
    )

    print("")
    print("=" * 72)
    print("Hmax SWEEP TEST")
    print("=" * 72)

    print("CSV :", csv_path)
    print(f"Samples : {len(H)}")
    print(f"fs : {fs:.6f} Hz")

    # H_major_max는 fitting 자체에는 영향을 주지 않는다.
    # 여기서는 fitting parameter를 한 번 얻기 위한 임시값.
    result = run_diagnosis(
        B_raw=B,
        H_raw=H,
        fs=fs,

        lower_bounds=cfg.JA_LOWER_BOUNDS,
        upper_bounds=cfg.JA_UPPER_BOUNDS,

        noise_frequencies=cfg.NOISE_FREQUENCIES,
        notch_q=cfg.NOTCH_Q,

        filter_h=cfg.FILTER_H,
        filter_b=cfg.FILTER_B,

        n_starts=cfg.N_STARTS,
        random_seed=cfg.RANDOM_SEED,
        max_nfev=cfg.MAX_NFEV,

        excitation_freq=excitation_freq,

        # fitting parameter를 얻기 위한 임시값
        H_major_max=0.18,

        major_points_per_cycle=cfg.MAJOR_POINTS_PER_CYCLE,
        major_cycles=cfg.MAJOR_CYCLES,
    )

    params = np.asarray(
        result["fitted_params"],
        dtype=float,
    )

    print("")
    print("[선택된 fitting]")
    print(
        "Selected cycle :",
        result.get("selected_cycle", "-"),
    )
    print(
        "NRMSE :",
        f"{result['fit_nrmse_percent']:.6f} %",
    )
    print(
        "Params [Ms, a, k, c, alpha] :"
    )
    print(params)

    return params


# ============================================================
# Hmax별 major-loop 생성
# ============================================================

def run_sweep(params):
    rows = []
    loops = {}

    for hmax in H_TEST_LIST:

        H_all = make_triangle_field(
            H_max=float(hmax),
            points_per_cycle=cfg.MAJOR_POINTS_PER_CYCLE,
            cycles=cfg.MAJOR_CYCLES,
        )

        B_all, M_all = simulate_ja(
            H=H_all,
            params=params,
            M0=0.0,
        )

        H_loop = np.asarray(
            H_all[
                -cfg.MAJOR_POINTS_PER_CYCLE:
            ],
            dtype=float,
        )

        B_loop = np.asarray(
            B_all[
                -cfg.MAJOR_POINTS_PER_CYCLE:
            ],
            dtype=float,
        )

        Bmax_abs = float(
            np.max(
                np.abs(B_loop)
            )
        )

        Bmax_pos = float(
            np.max(B_loop)
        )

        Bmin = float(
            np.min(B_loop)
        )

        area = float(
            calculate_loop_area(
                H_loop,
                B_loop,
            )
        )

        rows.append(
            {
                "Hmax": float(hmax),
                "Bmax_abs": Bmax_abs,
                "Bmax_pos": Bmax_pos,
                "Bmin": Bmin,
                "Area": area,
                "Bmax_ratio_percent": np.nan,
            }
        )

        loops[float(hmax)] = (
            H_loop,
            B_loop,
        )

    # sweep에서 관측된 최대 Bmax 대비 비율
    final_bmax = max(
        r["Bmax_abs"]
        for r in rows
    )

    for r in rows:
        if final_bmax > 0:
            r["Bmax_ratio_percent"] = (
                r["Bmax_abs"]
                / final_bmax
                * 100.0
            )

    return rows, loops


# ============================================================
# 결과 출력
# ============================================================

def print_table(rows):
    print("")
    print("=" * 90)
    print("Hmax SWEEP RESULT")
    print("=" * 90)

    print(
        f"{'Hmax':>10s} | "
        f"{'|B|max':>12s} | "
        f"{'B max':>12s} | "
        f"{'B min':>12s} | "
        f"{'Area':>14s} | "
        f"{'Bmax ratio':>11s}"
    )

    print("-" * 90)

    for r in rows:
        print(
            f"{r['Hmax']:10.2f} | "
            f"{r['Bmax_abs']:12.6f} | "
            f"{r['Bmax_pos']:12.6f} | "
            f"{r['Bmin']:12.6f} | "
            f"{r['Area']:14.6f} | "
            f"{r['Bmax_ratio_percent']:10.2f}%"
        )

    # sweep 최대 Bmax 기준 95%, 99% 도달 최초 지점
    for target in [95.0, 99.0]:
        hit = next(
            (
                r
                for r in rows
                if r["Bmax_ratio_percent"] >= target
            ),
            None,
        )

        if hit is not None:
            print(
                f"\nBmax가 sweep 최대값의 {target:.0f}%에 "
                f"처음 도달하는 Hmax ≈ {hit['Hmax']}"
            )


# ============================================================
# CSV 저장
# ============================================================

def save_result_csv(rows):
    out = Path(
        "hmax_sweep_results.csv"
    )

    fieldnames = [
        "Hmax",
        "Bmax_abs",
        "Bmax_pos",
        "Bmin",
        "Area",
        "Bmax_ratio_percent",
    ]

    with open(
        out,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\n결과 CSV 저장 :",
        out.resolve(),
    )


# ============================================================
# 그래프
# ============================================================

def draw_plots(rows, loops):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\nmatplotlib이 없어 그래프는 생략합니다."
        )
        return

    h_values = np.asarray(
        [
            r["Hmax"]
            for r in rows
        ],
        dtype=float,
    )

    b_values = np.asarray(
        [
            r["Bmax_abs"]
            for r in rows
        ],
        dtype=float,
    )

    area_values = np.asarray(
        [
            r["Area"]
            for r in rows
        ],
        dtype=float,
    )

    # 1) Hmax - Bmax
    plt.figure()

    plt.plot(
        h_values,
        b_values,
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Hmax")
    plt.ylabel("|B|max")
    plt.title("Hmax vs Bmax")
    plt.grid(True)
    plt.tight_layout()

    # 2) Hmax - Area
    plt.figure()

    plt.plot(
        h_values,
        area_values,
        marker="o",
    )

    plt.xscale("log")
    plt.xlabel("Hmax")
    plt.ylabel("Major-loop area")
    plt.title("Hmax vs major-loop area")
    plt.grid(True)
    plt.tight_layout()

    # 3) 대표 Hmax의 loop 모양 비교
    plt.figure()

    for hmax in PLOT_H_LIST:

        if float(hmax) not in loops:
            continue

        H_loop, B_loop = loops[
            float(hmax)
        ]

        plt.plot(
            H_loop,
            B_loop,
            label=f"Hmax={hmax:g}",
        )

    plt.xlabel("H")
    plt.ylabel("B")
    plt.title("Major-loop shape by Hmax")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# main
# ============================================================

def main():
    csv_path = find_csv()

    params = fit_once(
        csv_path
    )

    rows, loops = run_sweep(
        params
    )

    print_table(
        rows
    )

    save_result_csv(
        rows
    )

    draw_plots(
        rows,
        loops,
    )


if __name__ == "__main__":
    main()
