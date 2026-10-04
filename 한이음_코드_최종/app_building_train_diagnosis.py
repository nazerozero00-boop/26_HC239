"""구조물 피로도 진단 시스템 GUI — 건축물별 학습 모드 + 진단 모드.

실행: python app_building_train_diagnosis.py

학습 모드:
  건축물 선택 → 해당 건축물 CSV 등록/선택 → 실제 상태(안정/주의/위험) 선택 → J-A 특징 추출
  → SQLite 학습 이력 저장 → COMSOL 사전학습 데이터와 함께 MLP 재학습

진단 모드:
  UDP 측정 또는 파일 선택 → J-A 특징 추출 → 현재 active MLP 진단
  → 구조물명/측정지점과 함께 SQLite 영구 저장
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from diagnosis_engine import DiagnosisEngine
from measurement_csv_33hz import load_measurement
from ai.comsol_data import LABELS
from ai import sqlite_store
from ai.mlp_retraining import retrain_active_model, restore_base_model
from training_building_files import (
    add_csv_files, list_buildings, list_csv_files, resolve_csv, ensure_root
)

STATE_COLORS = ["#1D9E75", "#EF9F27", "#E24B4A"]
BG = "#F5F4EF"
CARD = "#FFFFFF"
MUTED = "#73726C"
TEXT = "#2C2C2A"
BLUE = "#378ADD"


class App:
    def __init__(self, root):
        self.root = root
        self.engine = DiagnosisEngine()
        self.engine.reset_session()
        self.workflow_mode = "diagnosis"
        self.selected = None
        self.selected_source_mode = "file"
        self.busy = False
        self.receiver = None
        self.measuring = False

        root.title("구조물 피로도 진단 - 건축물별 학습/진단")
        root.geometry("800x480")
        root.minsize(800, 480)
        root.configure(bg=BG)
        self._fonts()
        self._build()
        self.set_workflow_mode("diagnosis")

    def _fonts(self):
        self.f_title = tkfont.Font(family="DejaVu Sans", size=10, weight="bold")
        self.f_big = tkfont.Font(family="DejaVu Sans", size=20, weight="bold")
        self.f_lbl = tkfont.Font(family="DejaVu Sans", size=7)
        self.f_val = tkfont.Font(family="DejaVu Sans", size=10, weight="bold")
        self.f_small = tkfont.Font(family="DejaVu Sans", size=8)

    def _build(self):
        top = tk.Frame(self.root, bg=CARD, height=32)
        top.pack(fill="x", side="top")
        tk.Label(top, text="구조물 피로도 진단 시스템", bg=CARD, fg=TEXT,
                 font=self.f_title).pack(side="left", padx=(10, 8), pady=5)
        self.train_mode_btn = tk.Button(
            top, text="학습 모드", font=self.f_small, relief="flat", bd=0,
            command=lambda: self.set_workflow_mode("training"))
        self.train_mode_btn.pack(side="left", padx=2, pady=3, ipadx=5, ipady=1)
        self.diag_mode_btn = tk.Button(
            top, text="진단 모드", font=self.f_small, relief="flat", bd=0,
            command=lambda: self.set_workflow_mode("diagnosis"))
        self.diag_mode_btn.pack(side="left", padx=2, pady=3, ipadx=5, ipady=1)
        tk.Button(
            top, text="DB 이력", font=self.f_small, bg="#E8E6DF", fg=TEXT,
            relief="flat", bd=0, command=self.open_history
        ).pack(side="right", padx=(0, 8), pady=3, ipadx=5, ipady=1)
        self.model_status = tk.Label(top, text="MLP active", bg=CARD, fg=MUTED,
                                     font=self.f_small)
        self.model_status.pack(side="right", padx=6)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        left = tk.Frame(body, bg=BG, width=340)
        left.pack(side="left", fill="both", padx=8, pady=6)
        left.pack_propagate(False)

        # 구조물/측정 위치
        meta = tk.Frame(left, bg=CARD)
        meta.pack(fill="x", pady=(0, 5))
        self.structure_var = tk.StringVar()
        self.point_var = tk.StringVar()
        tk.Label(meta, text="구조물명", bg=CARD, fg=MUTED, font=self.f_lbl,
                 width=8, anchor="w").grid(row=0, column=0, padx=(6, 2), pady=3)
        tk.Entry(meta, textvariable=self.structure_var, font=self.f_small,
                 relief="solid", bd=1).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=3)
        tk.Label(meta, text="측정 지점", bg=CARD, fg=MUTED, font=self.f_lbl,
                 width=8, anchor="w").grid(row=1, column=0, padx=(6, 2), pady=3)
        tk.Entry(meta, textvariable=self.point_var, font=self.f_small,
                 relief="solid", bd=1).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=3)
        meta.grid_columnconfigure(1, weight=1)

        self.state_box = tk.Frame(left, bg="#E1F5EE", height=58)
        self.state_box.pack(fill="x", pady=(0, 5))
        self.state_box.pack_propagate(False)
        self.state_caption = tk.Label(self.state_box, text="현재 진단 상태", bg="#E1F5EE",
                                      fg=MUTED, font=self.f_lbl)
        self.state_caption.pack(pady=(5, 0))
        self.state_label = tk.Label(self.state_box, text="—", bg="#E1F5EE",
                                    fg=TEXT, font=self.f_big)
        self.state_label.pack()

        grid = tk.Frame(left, bg=BG)
        grid.pack(fill="x", pady=(0, 4))
        self.metric_vars = {}
        for i, key in enumerate(["Bmax", "Hmax", "Area", "dArea"]):
            cell = tk.Frame(grid, bg=CARD)
            cell.grid(row=i // 2, column=i % 2, sticky="nsew", padx=2, pady=2)
            grid.grid_columnconfigure(i % 2, weight=1)
            tk.Label(cell, text=key, bg=CARD, fg=MUTED, font=self.f_lbl).pack(
                anchor="w", padx=5, pady=(2, 0))
            val = tk.Label(cell, text="—", bg=CARD, fg=TEXT, font=self.f_val)
            val.pack(anchor="w", padx=5, pady=(0, 2))
            self.metric_vars[key] = val

        self.controls = tk.Frame(left, bg=BG)
        self.controls.pack(fill="x", pady=(0, 3))

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.prog_label = tk.Label(left, text="", bg=BG, fg=BLUE, font=self.f_small)
        self.answer_label = tk.Label(left, text="", bg=BG, fg=MUTED, font=self.f_small,
                                     justify="left", wraplength=325)
        self.answer_label.pack(fill="x", pady=(1, 3))

        pred = tk.Frame(left, bg=CARD)
        pred.pack(fill="x", side="bottom")
        tk.Label(pred, text="예상 위험 도달", bg=CARD, fg=MUTED,
                 font=self.f_lbl).pack(anchor="w", padx=7, pady=(3, 0))
        self.pred_label = tk.Label(pred, text="—", bg=CARD, fg="#E24B4A",
                                   font=self.f_val)
        self.pred_label.pack(anchor="w", padx=7, pady=(0, 3))

        right = tk.Frame(body, bg=CARD)
        right.pack(side="right", fill="both", expand=True, padx=(0, 8), pady=6)
        tk.Label(right, text="B-H 고리 (major-loop)", bg=CARD, fg=TEXT,
                 font=self.f_title).pack(anchor="w", padx=8, pady=(5, 1))
        self.bh_canvas = tk.Canvas(right, bg=CARD, highlightthickness=0, height=180)
        self.bh_canvas.pack(fill="x", padx=8)
        tk.Label(right, text="진단 Bmax 추이", bg=CARD, fg=TEXT,
                 font=self.f_title).pack(anchor="w", padx=8, pady=(3, 1))
        self.canvas = tk.Canvas(right, bg=CARD, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    def _clear_controls(self):
        for w in self.controls.winfo_children():
            w.destroy()

    def set_workflow_mode(self, mode):
        if self.busy or self.measuring:
            return
        self.workflow_mode = mode
        self._clear_controls()
        active_bg = BLUE
        inactive_bg = "#E8E6DF"
        self.train_mode_btn.config(
            bg=active_bg if mode == "training" else inactive_bg,
            fg="white" if mode == "training" else TEXT)
        self.diag_mode_btn.config(
            bg=active_bg if mode == "diagnosis" else inactive_bg,
            fg="white" if mode == "diagnosis" else TEXT)
        self.selected = None
        self.answer_label.config(text="")

        if mode == "training":
            self.state_caption.config(text="MLP 학습 상태")
            self.state_label.config(text="학습 대기", fg=TEXT)
            self.pred_label.config(text="—")
            self._build_training_controls()
        else:
            self.state_caption.config(text="현재 진단 상태")
            self.state_label.config(text="—", fg=TEXT)
            self._build_diagnosis_controls()

    def _build_training_controls(self):
        ensure_root()

        # 1) 건축물 선택 — 선택한 건축물에 따라 등록 CSV 목록이 분리된다.
        b_row = tk.Frame(self.controls, bg=BG)
        b_row.pack(fill="x", pady=(0, 2))
        tk.Label(b_row, text="학습 건축물", bg=BG, fg=MUTED,
                 font=self.f_small).pack(side="left")
        self.training_building_var = tk.StringVar(value="1번 건축물")
        self.training_building_combo = ttk.Combobox(
            b_row, textvariable=self.training_building_var,
            values=list_buildings(), state="readonly", width=15,
            font=self.f_small)
        self.training_building_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.training_building_combo.bind(
            "<<ComboboxSelected>>", self._on_training_building_changed)

        # 2) 선택한 건축물에 등록된 CSV만 표시한다.
        f_row = tk.Frame(self.controls, bg=BG)
        f_row.pack(fill="x", pady=(0, 2))
        tk.Label(f_row, text="등록 파일", bg=BG, fg=MUTED,
                 font=self.f_small).pack(side="left")
        self.training_file_var = tk.StringVar()
        self.training_file_combo = ttk.Combobox(
            f_row, textvariable=self.training_file_var,
            values=(), state="readonly", width=19,
            font=self.f_small)
        self.training_file_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.training_file_combo.bind(
            "<<ComboboxSelected>>", self._on_training_file_changed)

        add_row = tk.Frame(self.controls, bg=BG)
        add_row.pack(fill="x", pady=(0, 2))
        self.file_btn = tk.Button(
            add_row, text="선택 건축물에 CSV 추가", font=self.f_small,
            bg="#E8E6DF", fg=TEXT, relief="flat", command=self.on_add_training_files)
        self.file_btn.pack(side="left", fill="x", expand=True, ipady=2)
        tk.Button(
            add_row, text="목록 새로고침", font=self.f_small,
            bg="#E8E6DF", fg=TEXT, relief="flat",
            command=self._refresh_training_files
        ).pack(side="left", padx=(3, 0), ipady=2)

        self.file_label = tk.Label(
            self.controls, text="1번 건축물: 등록된 학습 파일 없음",
            bg=BG, fg=MUTED, font=self.f_small,
            wraplength=325, justify="left")
        self.file_label.pack(fill="x", pady=(0, 2))

        # 3) 파일의 실제 정답 라벨 지정
        row = tk.Frame(self.controls, bg=BG)
        row.pack(fill="x", pady=(0, 2))
        tk.Label(row, text="정답 라벨", bg=BG, fg=MUTED,
                 font=self.f_small).pack(side="left")
        self.training_label_var = tk.StringVar(value=LABELS[0])
        combo = ttk.Combobox(row, textvariable=self.training_label_var,
                             values=LABELS, state="readonly", width=8,
                             font=self.f_small)
        combo.pack(side="left", padx=5)
        tk.Label(row, text="(실제 상태)", bg=BG, fg=MUTED,
                 font=self.f_lbl).pack(side="left")

        self.action_btn = tk.Button(
            self.controls, text="선택 파일로 MLP 재학습", font=self.f_title,
            bg="#7A5AF8", fg="white", relief="flat", command=self.on_retrain)
        self.action_btn.pack(fill="x", ipady=4, pady=(0, 2))
        tk.Button(
            self.controls, text="COMSOL 기본 MLP 복원", font=self.f_small,
            bg="#E8E6DF", fg=TEXT, relief="flat", command=self.on_restore_base
        ).pack(fill="x", ipady=2)

        # 학습 건축물은 DB의 structure_name과 자동으로 맞춘다.
        self._on_training_building_changed()

    def _on_training_building_changed(self, _event=None):
        if self.workflow_mode != "training":
            return
        building = self.training_building_var.get().strip() or "1번 건축물"
        self.structure_var.set(building)
        self.selected = None
        self._refresh_training_files()

    def _refresh_training_files(self):
        if self.workflow_mode != "training":
            return
        building = self.training_building_var.get().strip() or "1번 건축물"
        try:
            files = list_csv_files(building)
        except Exception as e:
            self.file_label.config(text=f"파일 목록 오류: {e}", fg="#E24B4A")
            return
        self.training_file_combo.config(values=files)
        self.training_file_var.set("")
        self.selected = None
        if files:
            self.file_label.config(
                text=f"{building}: 학습 CSV {len(files)}개 등록됨", fg=MUTED)
        else:
            self.file_label.config(
                text=f"{building}: 등록된 학습 파일 없음", fg=MUTED)

    def on_add_training_files(self):
        if self.workflow_mode != "training" or self.busy:
            return
        building = self.training_building_var.get().strip() or "1번 건축물"
        paths = filedialog.askopenfilenames(
            title=f"{building}에 등록할 학습 CSV 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")])
        if not paths:
            return
        try:
            added = add_csv_files(building, paths)
            files = list_csv_files(building)
            self.training_file_combo.config(values=files)
            newest = os.path.basename(added[-1])
            self.training_file_var.set(newest)
            self._load_registered_training_file(building, newest)
            self.answer_label.config(
                text=f"{building}에 CSV {len(added)}개를 등록했습니다. 건축물별 폴더에 복사되어 관리됩니다.",
                fg="#1D9E75")
        except Exception as e:
            self.file_label.config(text=f"CSV 등록 실패: {e}", fg="#E24B4A")

    def _on_training_file_changed(self, _event=None):
        building = self.training_building_var.get().strip() or "1번 건축물"
        filename = self.training_file_var.get().strip()
        if filename:
            self._load_registered_training_file(building, filename)

    def _load_registered_training_file(self, building, filename):
        try:
            path = resolve_csv(building, filename)
            self.selected = load_measurement(path)
            self.selected_source_mode = "training_file"
            self.structure_var.set(building)
        except Exception as e:
            self.file_label.config(text=f"읽기 실패: {e}", fg="#E24B4A")
            self.selected = None
            return
        d = self.selected
        suffix = "33.1Hz fallback" if d.get("used_fallback_fs") else f"{d['fs']:.1f}Hz"
        info = f"{building} / {d['source_name']} / {len(d['H'])}점 / {suffix}"
        if d.get("input_kind") == "sensor_voltage":
            info += " / 센서 전압"
        self.file_label.config(text=info, fg="#1D9E75")
        if d.get("input_kind") == "sensor_voltage":
            self.answer_label.config(
                text="주의: 전압->H[A/m], B[T] 절대 환산 전 데이터입니다. "
                     "현재 학습/진단은 시스템 연동 검증용으로 해석하세요.",
                fg="#EF9F27")

    def _build_diagnosis_controls(self):
        self.meas_btn = tk.Button(
            self.controls, text="측정 시작 (UDP)", font=self.f_small,
            bg="#5FAE7E", fg="white", relief="flat", command=self.on_measure_toggle)
        self.meas_btn.pack(fill="x", ipady=3, pady=(0, 2))
        self.meas_label = tk.Label(self.controls, text="", bg=BG, fg=MUTED,
                                   font=self.f_small, wraplength=325)
        self.meas_label.pack(fill="x")
        self.file_btn = tk.Button(
            self.controls, text="진단 CSV 파일 선택", font=self.f_small,
            bg="#E8E6DF", fg=TEXT, relief="flat", command=self.on_pick_file)
        self.file_btn.pack(fill="x", ipady=3, pady=(0, 2))
        self.file_label = tk.Label(self.controls, text="선택된 진단 파일 없음",
                                   bg=BG, fg=MUTED, font=self.f_small,
                                   wraplength=325, justify="left")
        self.file_label.pack(fill="x", pady=(0, 3))
        self.action_btn = tk.Button(
            self.controls, text="진단 실행", font=self.f_title,
            bg=BLUE, fg="white", relief="flat", command=self.on_diagnose)
        self.action_btn.pack(fill="x", ipady=5)

    def _location(self):
        structure = self.structure_var.get().strip() or "미입력"
        point = self.point_var.get().strip() or "미입력"
        return structure, point

    def on_pick_file(self):
        """진단 모드에서 외부 CSV 1개를 직접 선택한다."""
        if self.workflow_mode != "diagnosis":
            return
        path = filedialog.askopenfilename(
            title="진단할 측정 CSV 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            self.selected = load_measurement(path)
            self.selected_source_mode = "file"
        except Exception as e:
            self.file_label.config(text=f"읽기 실패: {e}", fg="#E24B4A")
            self.selected = None
            return
        d = self.selected
        suffix = "33.1Hz fallback" if d.get("used_fallback_fs") else f"{d['fs']:.1f}Hz"
        info = f"{d['source_name']} / {len(d['H'])}점 / {suffix}"
        if d.get("input_kind") == "sensor_voltage":
            info += " / 센서 전압"
        self.file_label.config(text=info, fg="#1D9E75")
        if d.get("input_kind") == "sensor_voltage":
            self.answer_label.config(
                text="주의: 전압->H[A/m], B[T] 절대 환산 전 데이터입니다. "
                     "현재 학습/진단은 시스템 연동 검증용으로 해석하세요.",
                fg="#EF9F27")

    # ---------- UDP 측정 ----------
    def on_measure_toggle(self):
        if self.workflow_mode != "diagnosis" or self.busy:
            return
        if not self.measuring:
            from measurement_udp_receiver import MeasurementReceiver
            self.measuring = True
            self.meas_btn.config(text="측정 정지 및 저장", bg="#E24B4A")
            self.meas_label.config(text="측정 준비 중...", fg=BLUE)

            def on_status(msg):
                self.root.after(0, lambda: self.meas_label.config(text=msg, fg=BLUE))

            self.receiver = MeasurementReceiver(on_status=on_status)
            self.receiver.start()
        else:
            self.measuring = False
            self.meas_btn.config(text="측정 시작 (UDP)", bg="#5FAE7E")
            self.meas_label.config(text="저장 중...", fg=MUTED)

            def stop_worker():
                try:
                    path, n = self.receiver.stop()
                    err = None
                except Exception as e:
                    path, n, err = None, 0, str(e)
                self.root.after(0, lambda: self._after_measure(path, n, err))
            threading.Thread(target=stop_worker, daemon=True).start()

    def _after_measure(self, path, n, err=None):
        if err:
            self.meas_label.config(text=f"측정 종료 오류: {err}", fg="#E24B4A")
            return
        if path is None:
            self.meas_label.config(text=f"저장 실패 (수신 {n}점)", fg="#E24B4A")
            return
        try:
            self.selected = load_measurement(path)
            self.selected_source_mode = "udp"
        except Exception as e:
            self.meas_label.config(text=f"저장됐으나 로드 실패: {e}", fg="#E24B4A")
            return
        self.meas_label.config(text=f"저장 완료: {os.path.basename(path)} ({n}점)", fg="#1D9E75")
        self.file_label.config(
            text=f"자동 선택됨 / {len(self.selected['H'])}점 / {self.selected['fs']:.1f}Hz",
            fg="#1D9E75")

    # ---------- 진단 ----------
    def on_diagnose(self):
        if self.busy or self.workflow_mode != "diagnosis":
            return
        if self.selected is None:
            self.answer_label.config(text="먼저 측정하거나 진단 CSV를 선택하세요.", fg="#E24B4A")
            return
        self._start_busy("J-A fitting + MLP 진단 중...")
        threading.Thread(target=self._diagnose_worker, daemon=True).start()

    def _diagnose_worker(self):
        try:
            d = self.selected
            # 수동 파일은 독립 측정으로, UDP 연속 측정은 dArea 상태 유지
            if self.selected_source_mode != "udp":
                self.engine.reset_classifier_state()
            structure, point = self._location()
            r = self.engine.run_once(
                d["time"], d["H"], d["B"], fs=d["fs"],
                structure_name=structure,
                measurement_point=point,
                source_mode=self.selected_source_mode,
            )
            if d.get("k") is not None:
                r["k"] = d["k"]
            true_label = d.get("true_label")
        except Exception as e:
            r, true_label = {"_error": str(e)}, None
        self.root.after(0, lambda: self._finish_diagnose(r, true_label))

    def _finish_diagnose(self, r, true_label):
        self._stop_busy()
        if "_error" in r:
            self.state_label.config(text="오류", fg="#E24B4A")
            self.answer_label.config(text=f"진단 오류: {r['_error']}", fg="#E24B4A")
            return
        self._update_diagnosis_ui(r, true_label)

    # ---------- 학습 ----------
    def on_retrain(self):
        if self.busy or self.workflow_mode != "training":
            return
        if self.selected is None:
            self.answer_label.config(text="건축물을 선택하고 등록된 학습 CSV를 선택하세요.", fg="#E24B4A")
            return
        label = self.training_label_var.get()
        if label not in LABELS:
            self.answer_label.config(text="정답 라벨을 선택하세요.", fg="#E24B4A")
            return
        self._start_busy("J-A 특징 추출 + MLP 재학습 중...")
        threading.Thread(target=self._retrain_worker, args=(label,), daemon=True).start()

    def _retrain_worker(self, label):
        try:
            d = self.selected
            feat = self.engine.extract_features(d["time"], d["H"], d["B"], fs=d["fs"])
            label_idx = LABELS.index(label)
            structure, point = self._location()
            sqlite_store.insert_training_sample(
                source_file=d.get("source_name", "unknown.csv"),
                label=label,
                label_idx=label_idx,
                bmax=feat["Bmax"], hmax=feat["Hmax"], area=feat["Area"],
                nrmse=feat.get("nrmse"),
                structure_name=structure, measurement_point=point,
                input_kind=d.get("input_kind", "unknown"),
            )
            training_records = sqlite_store.load_training_samples()
            report = retrain_active_model(training_records)
            self.engine.reload_model()
            payload = {"feature": feat, "report": report, "label": label,
                       "input_kind": d.get("input_kind")}
        except Exception as e:
            payload = {"_error": str(e)}
        self.root.after(0, lambda: self._finish_retrain(payload))

    def _finish_retrain(self, payload):
        self._stop_busy()
        if "_error" in payload:
            self.state_label.config(text="학습 오류", fg="#E24B4A")
            self.answer_label.config(text=f"재학습 오류: {payload['_error']}", fg="#E24B4A")
            return
        feat = payload["feature"]
        report = payload["report"]
        self.state_label.config(text="학습 완료", fg="#7A5AF8")
        self.metric_vars["Bmax"].config(text=f"{feat['Bmax']:.3f}")
        self.metric_vars["Hmax"].config(text=f"{feat['Hmax']:.1f}")
        self.metric_vars["Area"].config(text=f"{feat['Area']:.1f}")
        self.metric_vars["dArea"].config(text="—")
        self._draw_bh(feat["major_loop"], "#7A5AF8", f"학습:{payload['label']}")
        msg = (f"{self.structure_var.get().strip() or '미입력'} 재학습 완료 / 실측 학습파일 {report['real_files']}개 누적 / "
               f"COMSOL 검증 정확도 {report['comsol_validation_accuracy']*100:.1f}%")
        if payload.get("input_kind") == "sensor_voltage":
            msg += "\n※ 센서 전압 절대 환산 전이므로 물리적 진단 정확도 검증은 별도 필요"
        self.answer_label.config(text=msg, fg="#1D9E75")
        self.model_status.config(text="MLP 재학습 적용")

    def on_restore_base(self):
        if self.busy:
            return
        try:
            restore_base_model()
            self.engine.reload_model()
            self.engine.reset_classifier_state()
            self.model_status.config(text="COMSOL 기본 MLP")
            self.state_label.config(text="기본모델", fg=BLUE)
            self.answer_label.config(
                text="COMSOL 사전학습 MLP를 active 모델로 복원했습니다. 학습 이력 DB는 유지됩니다.",
                fg="#1D9E75")
        except Exception as e:
            self.answer_label.config(text=f"기본모델 복원 실패: {e}", fg="#E24B4A")

    def _start_busy(self, text):
        self.busy = True
        if hasattr(self, "action_btn"):
            self.action_btn.config(state="disabled")
        self.progress.pack(fill="x", pady=(0, 1))
        self.prog_label.config(text=text)
        self.prog_label.pack(fill="x")
        self.progress.start(12)

    def _stop_busy(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.prog_label.pack_forget()
        if hasattr(self, "action_btn"):
            self.action_btn.config(state="normal")
        self.busy = False

    # ---------- SQLite 이력 ----------
    def open_history(self):
        win = tk.Toplevel(self.root)
        win.title("SQLite 진단/학습 이력")
        win.geometry("790x430")
        win.configure(bg=BG)
        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=7, pady=7)

        diag_tab = tk.Frame(notebook, bg=BG)
        train_tab = tk.Frame(notebook, bg=BG)
        notebook.add(diag_tab, text="진단 이력")
        notebook.add(train_tab, text="학습 이력")

        self._build_history_table(
            diag_tab,
            columns=("id", "time", "structure", "point", "Bmax", "Hmax", "Area", "dArea", "label"),
            headings=("ID", "시각", "구조물", "측정지점", "Bmax", "Hmax", "Area", "dArea", "결과"),
            loader=self.engine.get_saved_history,
            formatter=self._format_diag_row,
        )
        self._build_history_table(
            train_tab,
            columns=("id", "time", "file", "structure", "point", "Bmax", "Hmax", "Area", "label"),
            headings=("ID", "시각", "학습파일", "구조물", "측정지점", "Bmax", "Hmax", "Area", "라벨"),
            loader=self.engine.get_training_history,
            formatter=self._format_train_row,
        )

    def _build_history_table(self, parent, columns, headings, loader, formatter):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, padx=5, pady=5)
        tree = ttk.Treeview(wrap, columns=columns, show="headings")
        for col, head in zip(columns, headings):
            width = 120 if col in ("time", "file", "structure", "point") else 70
            tree.heading(col, text=head)
            tree.column(col, width=width, minwidth=45, anchor="center")
        ys = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        status = tk.StringVar()
        footer = tk.Frame(parent, bg=BG)
        footer.pack(fill="x", padx=7, pady=(0, 5))
        tk.Label(footer, textvariable=status, bg=BG, fg=MUTED,
                 font=self.f_small).pack(side="left")

        def refresh():
            for item in tree.get_children():
                tree.delete(item)
            try:
                records = loader()
                for r in records:
                    tree.insert("", "end", values=formatter(r))
                status.set(f"총 {len(records)}건 / 최신 기록이 위에 표시됩니다.")
            except Exception as e:
                status.set(f"DB 오류: {e}")
        tk.Button(footer, text="새로고침", font=self.f_small, bg="#E8E6DF",
                  fg=TEXT, relief="flat", command=refresh).pack(side="right")
        refresh()

    @staticmethod
    def _time_text(ts):
        from datetime import datetime
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")

    def _format_diag_row(self, r):
        return (
            r["diagnosis_id"], self._time_text(r["timestamp"]),
            r["structure_name"], r["measurement_point"],
            f"{r['Bmax']:.3f}", f"{r['Hmax']:.1f}", f"{r['Area']:.1f}",
            f"{r['dArea']:+.1f}", r["label"])

    def _format_train_row(self, r):
        return (
            r["training_id"], self._time_text(r["timestamp"]), r["source_file"],
            r["structure_name"], r["measurement_point"],
            f"{r['Bmax']:.3f}", f"{r['Hmax']:.1f}", f"{r['Area']:.1f}", r["label"])

    # ---------- UI drawing ----------
    def _update_diagnosis_ui(self, r, true_label=None):
        idx = r["label_idx"]
        if idx < 0:
            color, tint = MUTED, "#ECEBE6"
        else:
            color = STATE_COLORS[idx]
            tint = ["#E1F5EE", "#FBEEDA", "#FCEBEB"][idx]
        self.state_box.config(bg=tint)
        for w in self.state_box.winfo_children():
            w.config(bg=tint)

        if idx < 0:
            self.state_label.config(text="철근 없음", fg=color)
            self.answer_label.config(text=f"신호 약함 (Bmax={r['Bmax']:.3f})", fg=color)
            self._set_metrics(r, no_darea=True)
            self.pred_label.config(text="—")
        else:
            self.state_label.config(text=r["label"], fg=color)
            if true_label is not None:
                ok = r["label"] == true_label
                self.answer_label.config(
                    text=f"정답 {true_label} / {'일치' if ok else '불일치'}",
                    fg="#1D9E75" if ok else "#E24B4A")
            else:
                note = f"J-A NRMSE {r.get('nrmse', 0):.1f}% / fs {r.get('fs', 0):.1f}Hz"
                self.answer_label.config(text=note, fg=MUTED)
            self._set_metrics(r)
            m = r.get("months_to_risk")
            self.pred_label.config(text=f"약 {m}개월 후" if m is not None else "추정 중...")
        if r.get("major_loop"):
            self._draw_bh(r["major_loop"], color, r["label"])
        self._draw_trend(self.engine.get_history())

    def _set_metrics(self, r, no_darea=False):
        self.metric_vars["Bmax"].config(text=f"{r['Bmax']:.3f}")
        self.metric_vars["Hmax"].config(text=f"{r['Hmax']:.1f}")
        self.metric_vars["Area"].config(text=f"{r['Area']:.1f}")
        if no_darea:
            self.metric_vars["dArea"].config(text="—")
        else:
            d = r["dArea"]
            self.metric_vars["dArea"].config(text=f"{d:+.1f}")

    def _draw_bh(self, loop, color, label):
        import numpy as np
        H, B = loop
        H = np.asarray(H); B = np.asarray(B)
        c = self.bh_canvas
        c.delete("all"); c.update_idletasks()
        W = max(c.winfo_width(), 430); Hgt = 180; pad = 16
        hmin, hmax = float(H.min()), float(H.max())
        bmin, bmax = float(B.min()), float(B.max())
        hr = (hmax - hmin) or 1.0; br = (bmax - bmin) or 1.0
        X = lambda h: pad + (W - 2*pad) * (h - hmin) / hr
        Y = lambda b: Hgt - pad - (Hgt - 2*pad) * (b - bmin) / br
        if bmin <= 0 <= bmax:
            c.create_line(pad, Y(0), W-pad, Y(0), fill="#D8D6CE")
        if hmin <= 0 <= hmax:
            c.create_line(X(0), pad, X(0), Hgt-pad, fill="#D8D6CE")
        step = max(1, len(H)//1200)
        pts = []
        for h, b in zip(H[::step], B[::step]):
            pts.extend([X(float(h)), Y(float(b))])
        if len(pts) >= 4:
            c.create_line(*pts, fill=color, width=2, smooth=True)
        c.create_text(W-pad, pad+4, text=f"B-H / {label}", anchor="e",
                      fill=MUTED, font=self.f_small)

    def _draw_trend(self, hist):
        c = self.canvas
        c.delete("all"); c.update_idletasks()
        W = max(c.winfo_width(), 430); Hgt = max(c.winfo_height(), 110); pad = 22
        if not hist:
            return
        vals = [float(h["Bmax"]) for h in hist]
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            vmin -= 0.5; vmax += 0.5
        margin = 0.08 * (vmax-vmin)
        vmin -= margin; vmax += margin
        n = len(vals)
        X = lambda i: pad + (W - 2*pad) * i / max(n-1, 1)
        Y = lambda v: Hgt-pad-(Hgt-2*pad)*(v-vmin)/(vmax-vmin)
        c.create_line(pad, Hgt-pad, W-pad, Hgt-pad, fill="#B4B2A9")
        c.create_line(pad, pad, pad, Hgt-pad, fill="#B4B2A9")
        pts = [(X(i), Y(v)) for i, v in enumerate(vals)]
        for i in range(len(pts)-1):
            c.create_line(*pts[i], *pts[i+1], fill=BLUE, width=2)
        for i, (px, py) in enumerate(pts):
            idx = hist[i].get("label_idx", -1)
            col = STATE_COLORS[idx] if idx in (0, 1, 2) else MUTED
            c.create_oval(px-3, py-3, px+3, py+3, fill=col, outline="")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
