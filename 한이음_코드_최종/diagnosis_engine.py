"""통합 엔진 — 33.1 Hz/UDP, H-only notch, J-A, MLP, SQLite.

진단: 측정 → FFT → H notch(B RAW) → J-A fitting → major-loop → 특징 → MLP
학습: 측정 파일 → 동일 특징 추출 → 사용자 라벨과 함께 DB 저장 → MLP 재학습
"""
from __future__ import annotations

import numpy as np

from pipeline import run_diagnosis
from ai.mlp_model import Diagnoser
from ai.predict import predict_time_to_risk
from ai import session_history, sqlite_store
import ja_config as cfg


def features_from_major_loop(H_major, B_major, major_area):
    Bmax = float(np.max(np.abs(B_major)))
    Hmax = float(np.max(np.abs(H_major)))
    Area = float(major_area)
    return Bmax, Hmax, Area


class DiagnosisEngine:
    REBAR_BMAX_MIN = cfg.REBAR_BMAX_MIN

    def __init__(self, model_path=None):
        sqlite_store.init_db()
        self.diagnoser = Diagnoser(model_path=model_path)

    def _resolve_fs(self, time_data, fs):
        if fs is not None:
            return float(fs)
        if time_data is None:
            return float(cfg.DEFAULT_SAMPLING_FREQUENCY)
        from fft_analysis import estimate_sampling_frequency
        try:
            return float(estimate_sampling_frequency(time_data))
        except (TypeError, ValueError):
            return float(cfg.DEFAULT_SAMPLING_FREQUENCY)

    def extract_features(self, time_data, H_raw, B_raw, fs=None,
                         noise_frequencies=None):
        """진단/학습 공통 J-A 특징 추출. 이 함수 자체는 AI 진단/DB 저장을 하지 않는다."""
        H_raw = np.asarray(H_raw, dtype=float)
        B_raw = np.asarray(B_raw, dtype=float)
        fs = self._resolve_fs(time_data, fs)

        excitation_freq = (
            fs * float(cfg.ARDUINO_PHASE_STEP_RAD) / (2.0 * np.pi)
        )

        nf = list(cfg.NOISE_FREQUENCIES if noise_frequencies is None else noise_frequencies)
        # 실측 fs가 예상보다 낮아져도 Nyquist 밖 notch 때문에 전체 진단이 죽지 않게 제외.
        nyquist = fs / 2.0
        nf = [float(f) for f in nf if 0 < float(f) < nyquist]

        result = run_diagnosis(
            B_raw=B_raw,
            H_raw=H_raw,
            fs=fs,
            lower_bounds=cfg.JA_LOWER_BOUNDS,
            upper_bounds=cfg.JA_UPPER_BOUNDS,
            noise_frequencies=nf,
            notch_q=cfg.NOTCH_Q,
            filter_h=cfg.FILTER_H,
            filter_b=cfg.FILTER_B,
            n_starts=cfg.N_STARTS,
            random_seed=cfg.RANDOM_SEED,
            max_nfev=cfg.MAX_NFEV,
            H_major_max=cfg.H_MAJOR_MAX,
            major_points_per_cycle=cfg.MAJOR_POINTS_PER_CYCLE,
            major_cycles=cfg.MAJOR_CYCLES,
            excitation_freq=excitation_freq,
        )

        Bmax, Hmax, Area = features_from_major_loop(
            result["H_major"], result["B_major"], result["major_area"]
        )
        return {
            "Bmax": Bmax,
            "Hmax": Hmax,
            "Area": Area,
            "major_loop": (result["H_major"], result["B_major"]),
            "nrmse": float(result["fit_nrmse_percent"]),
            "fs": fs,
            "excitation_freq": excitation_freq,
            "noise_frequencies_used": nf,
        }

    def run_once(self, time_data, H_raw, B_raw, fs=None,
                 noise_frequencies=None, structure_name="",
                 measurement_point="", source_mode="file"):
        feat = self.extract_features(
            time_data, H_raw, B_raw, fs=fs, noise_frequencies=noise_frequencies
        )
        out = self._diagnose_features(
            feat["Bmax"], feat["Hmax"], feat["Area"],
            structure_name=structure_name,
            measurement_point=measurement_point,
            source_mode=source_mode,
        )
        out["major_loop"] = feat["major_loop"]
        out["nrmse"] = feat["nrmse"]
        out["fs"] = feat["fs"]
        out["excitation_freq"] = feat["excitation_freq"]
        return out

    def run_once_features(self, Bmax, Hmax, Area, structure_name="",
                          measurement_point="", source_mode="feature_test"):
        return self._diagnose_features(
            Bmax, Hmax, Area,
            structure_name=structure_name,
            measurement_point=measurement_point,
            source_mode=source_mode,
        )

    def _diagnose_features(self, Bmax, Hmax, Area, structure_name="",
                           measurement_point="", source_mode="unknown"):
        if Bmax < self.REBAR_BMAX_MIN:
            sqlite_store.insert_diagnosis(
                Bmax, Hmax, Area, 0.0, "철근 없음", -1,
                structure_name=structure_name,
                measurement_point=measurement_point,
                source_mode=source_mode,
            )
            return {
                "Bmax": Bmax, "Hmax": Hmax, "Area": Area, "dArea": 0.0,
                "label": "철근 없음", "label_idx": -1, "rebar": False,
                "months_to_risk": None, "probability": 0.0, "trend": "stable",
                "major_loop": None, "nrmse": None,
            }

        diag = self.diagnoser.diagnose(Bmax, Hmax, Area)
        label, idx, dArea = diag["label"], diag["index"], diag["dArea"]
        session_history.append_record(
            Bmax, Hmax, Area, dArea, label, idx,
            structure_name=structure_name,
            measurement_point=measurement_point,
            source_mode=source_mode,
        )
        pred = predict_time_to_risk(session_history.area_series())
        return {
            "Bmax": Bmax, "Hmax": Hmax, "Area": Area, "dArea": dArea,
            "label": label, "label_idx": idx, "rebar": True,
            "months_to_risk": pred["months_to_risk"],
            "probability": pred["probability"], "trend": pred["trend"],
            "major_loop": None, "nrmse": None,
        }

    def reload_model(self):
        self.diagnoser.reload_model()
        self.diagnoser.reset()

    def get_history(self):
        return session_history.load_history()

    def get_saved_history(self, limit=None):
        return sqlite_store.load_diagnosis_records(limit=limit)

    def get_training_history(self, limit=None):
        return sqlite_store.load_training_samples(limit=limit)

    def reset_classifier_state(self):
        self.diagnoser.reset()

    def reset_session(self):
        session_history.clear_history()
        self.diagnoser.reset()
