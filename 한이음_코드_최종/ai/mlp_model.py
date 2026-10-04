"""활성 MLP 모델 로드/추론.

초기 active 모델은 COMSOL 사전학습 MLP이며, 학습 모드 재학습 후 같은 경로의
model_active_mlp_2feat.pkl이 갱신된다.
"""
from __future__ import annotations

import joblib
import numpy as np
from ai.comsol_data import LABELS
from ai.mlp_retraining import ACTIVE_MODEL_PATH, ensure_models

_MODEL = None
_MODEL_PATH_LOADED = None


def load_model(path=None, force_reload=False):
    global _MODEL, _MODEL_PATH_LOADED
    ensure_models()
    target = path or ACTIVE_MODEL_PATH
    if force_reload or _MODEL is None or _MODEL_PATH_LOADED != target:
        _MODEL = joblib.load(target)
        _MODEL_PATH_LOADED = target
    return _MODEL


def reload_active_model():
    return load_model(ACTIVE_MODEL_PATH, force_reload=True)


class Diagnoser:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = load_model(model_path)
        self.prev_area = None

    def reload_model(self):
        self.model = load_model(self.model_path or ACTIVE_MODEL_PATH, force_reload=True)

    def diagnose(self, Bmax, Hmax, Area):
        """
        현재 프로토타입 분류 입력:
            [Bmax, Area]

        Hmax 인자는 diagnosis_engine.py와의 호환 때문에 유지하지만,
        현재 프로토타입에서는 J-A 복원 범위 H_MAJOR_MAX=10000으로
        고정되어 있으므로 분류 입력에는 사용하지 않는다.

        실제 시스템에서 절대 H를 측정/보정하여 Hmax가
        상태에 따라 달라지는 독립 특징이 되면,
        별도 재학습 후 다시 입력으로 추가할 수 있다.
        """
        dArea = (
            0.0
            if self.prev_area is None
            else float(
                Area - self.prev_area
            )
        )

        self.prev_area = float(
            Area
        )

        x = np.array(
            [[
                Bmax,
                Area,
            ]],
            dtype=float,
        )

        idx = int(
            self.model.predict(
                x
            )[0]
        )

        return {
            "label": LABELS[idx],
            "index": idx,
            "dArea": dArea,
        }

    def reset(self):
        self.prev_area = None
