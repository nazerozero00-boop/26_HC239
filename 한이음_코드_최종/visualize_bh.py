"""B-H 히스테리시스 고리 시각화 (독립 실행 + GUI 임베드 겸용).

두 가지로 쓸 수 있다:
  1) 독립 실행: python visualize_bh.py
     → COMSOL 각 k의 실제 고리를 그려 PNG로 저장 (시험평가/발표용)
  2) GUI 임베드: draw_bh_loop(canvas, H, B, ...) 함수를 app_building_train_diagnosis.py에서 활용 가능
     → Tkinter Canvas에 실시간 고리를 그림 (matplotlib 불필요, 가벼움)

라즈베리파이 부담 최소화를 위해, GUI용 함수는 순수 Tkinter Canvas만 쓴다.
독립 실행(PNG 저장)만 matplotlib을 쓴다.
"""
import numpy as np


# ── GUI 임베드용: 순수 Tkinter Canvas 렌더 (matplotlib 없음) ──────────
def draw_bh_loop(canvas, H, B, title="B-H 고리", area=None,
                 color="#378ADD", fill=True):
    """Tkinter Canvas에 B-H 히스테리시스 고리를 그린다.

    canvas: tkinter.Canvas
    H, B  : 같은 길이의 배열 (H=자기장, B=자속밀도)
    area  : 표시할 면적값 (None이면 생략)
    라즈베리파이에서도 가볍게 도는 순수 캔버스 드로잉.
    """
    canvas.delete("all")
    canvas.update_idletasks()
    W = canvas.winfo_width() or 400
    Ht = canvas.winfo_height() or 300
    pad = 44

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)
    if len(H) < 2:
        return

    hmin, hmax = float(np.min(H)), float(np.max(H))
    bmin, bmax = float(np.min(B)), float(np.max(B))
    # 0을 항상 포함(사분면 축 보이게)
    hmin, hmax = min(hmin, 0), max(hmax, 0)
    bmin, bmax = min(bmin, 0), max(bmax, 0)
    hrange = (hmax - hmin) or 1.0
    brange = (bmax - bmin) or 1.0

    def sx(h):
        return pad + (W - 2 * pad) * (h - hmin) / hrange

    def sy(b):
        return Ht - pad - (Ht - 2 * pad) * (b - bmin) / brange

    # 축 (0선)
    x0, y0 = sx(0), sy(0)
    canvas.create_line(pad, y0, W - pad, y0, fill="#B4B2A9")   # H축
    canvas.create_line(x0, pad, x0, Ht - pad, fill="#B4B2A9")  # B축
    canvas.create_text(W - pad + 4, y0 + 10, text="H", anchor="w",
                       fill="#73726C", font=("DejaVu Sans", 9))
    canvas.create_text(x0 + 6, pad - 6, text="B", anchor="w",
                       fill="#73726C", font=("DejaVu Sans", 9))

    # 채우기 (고리 내부)
    pts = []
    for h, b in zip(H, B):
        pts.extend([sx(h), sy(b)])
    if fill and len(pts) >= 6:
        canvas.create_polygon(pts, fill="#DCE9F8", outline="", stipple="gray25")

    # 고리 선
    for i in range(len(H) - 1):
        canvas.create_line(sx(H[i]), sy(B[i]), sx(H[i+1]), sy(B[i+1]),
                           fill=color, width=2)
    # 시작점 표시
    canvas.create_oval(sx(H[0])-4, sy(B[0])-4, sx(H[0])+4, sy(B[0])+4,
                       fill="#E24B4A", outline="")

    # 제목 + 면적
    canvas.create_text(W/2, 16, text=title, fill="#2C2C2A",
                       font=("DejaVu Sans", 11, "bold"))
    if area is not None:
        canvas.create_text(W - pad, Ht - pad + 14,
                           text=f"면적 = {area:.0f}", anchor="e",
                           fill="#185FA5", font=("DejaVu Sans", 10, "bold"))


# ── 독립 실행용: COMSOL 실제 고리를 PNG로 (matplotlib) ────────────────
def _parse_comsol_loops():
    """txt6.txt를 파싱해 k별 (H, B) 고리 시계열을 반환."""
    import re
    from collections import OrderedDict
    RAW = "/mnt/user-data/uploads/txt6.txt"
    with open(RAW, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    header, data = None, None
    for ln in lines:
        s = ln.strip()
        if s.startswith("% x") and "mf.Bx" in s:
            header = s
        elif s and not s.startswith("%"):
            data = s
    specs = [(m.group(1), int(m.group(2))) for m in
             re.finditer(r'mf\.(Bx|Hx)\s*\([^)]*\)\s*@\s*t=[\d.]+,\s*k_fatigue=(\d+)', header)]
    vals = [float(x) for x in data.split()][2:]  # x,y 제외
    kd = OrderedDict()
    for (typ, k), v in zip(specs, vals):
        kd.setdefault(k, {"Bx": [], "Hx": []})[typ].append(v)
    return kd


def save_comsol_gallery(out="comsol_bh_gallery.png"):
    """COMSOL 모든 k의 고리를 한 장에 그려 저장 (시험평가/발표용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

    import sys
    sys.path.insert(0, ".")
    from features import loop_area
    sys.path.insert(0, "ai")
    from comsol_data import label_from_k, LABELS

    kd = _parse_comsol_loops()
    ks = list(kd.keys())
    ncol = 4
    nrow = int(np.ceil(len(ks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3.2*nrow))
    axes = np.array(axes).reshape(-1)
    colors = {0: "#1D9E75", 1: "#EF9F27", 2: "#E24B4A"}

    for ax, k in zip(axes, ks):
        H = np.array(kd[k]["Hx"]); B = np.array(kd[k]["Bx"])
        n = min(len(H), len(B)); H, B = H[:n], B[:n]
        area = loop_area(np.column_stack([H, B]))
        lab = label_from_k(k)
        c = colors[lab]
        ax.fill(H, B, alpha=0.15, color=c)
        ax.plot(H, B, "-", color=c, lw=1.5)
        ax.plot(H[0], B[0], "o", color="#E24B4A", ms=4)
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
        ax.set_title(f"k={k}  [{LABELS[lab]}]  면적={area:.0f}", fontsize=10)
        ax.set_xlabel("H (A/m)", fontsize=8); ax.set_ylabel("B (T)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.25)

    for ax in axes[len(ks):]:
        ax.axis("off")
    fig.suptitle("COMSOL B-H 히스테리시스 고리 (피로계수 k별)", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"저장: {out}")
    return out


if __name__ == "__main__":
    save_comsol_gallery()
