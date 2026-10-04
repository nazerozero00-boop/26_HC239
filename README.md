# 26_HC239

1.프로젝트 개요 
1-1.프로젝트 소개
프로젝트 명 : 안전등급 미달 시설물을 위한 B-H커브의 면적 변화량 비파괴 계측 기반 피로도 분류 시스템
프로젝트 정의 : 철근의 자기 특성 변화를 분석하여 구조물 내부 철근의 피로 상태를 비파괴적으로 진단하는 AI 기반 상태진단 시스템

1-2.개발 배경 및 필요성
최근 노후된 인프라(건물, 교량 등)의 증가로 구조물의 지속적인 안전 관리와 상태진단의 중요성이 커지고 있습니다. 콘크리트 내부 철근은 반복 하중으로 인하여 피로가 누적될 수 있지만, 외부에서 직접 상태를 확인하기 어렵습니다. 기존의 구조물 피로도 검사 방식은 전문 장비와 많은 비용이 필요하고, 주로 균열이나 결함이 발생한 이후의 손상을 확인하는 방식이 많아서 초기 피로 누적을 사전에 파악하기 어렵습니다. 따라서 균열이나 결함이 발생하기 전에 철근의 피로 상태를 조기에 파악하고, 초기 정비를 진행할 수 있게끔 정량적 지표를 제공하는 진단 기술이 필요합니다.

1-3.프로젝트 특장점
- 균열 발생 이후가 아닌, 철근의 B-H곡선 특성 변화를 활용하여 피로 상태 변화를 조기에 파악
- 센서로 자기 응답을 계측하고 특징값을 기반으로 상태를 판단
- 측정 데이터를 자동으로 분석하여 철근의 상태를 3단계로 분류하기 때문에 전문가 없이도 진단이 가능함 
- 반복된 측정과 진단 이력 관리를 통해 구조물의 상태 변화를 추적하는 것에 활용
- 구조물의 공사 과정에서 철근의 존재 여부를 판단하여 부실공사를 방지  

1-4.주요 기능
- 철근 상태 자동 진단 : 측정 데이터를 분석하여 철근의 상태를 안정/주의/위험 3단계로 분류
- 실시간 측정 데이터 수집 : 센싱 디바이스에서 측정한 데이터를 실시간으로 수신하고 자동으로 저장
- 진단 결과 시각화 : B-H 곡선과 주요 특징값, 진단 결과를 GUI에서 한눈에 확인 가능
- 건축물 별 진단 이력 관리 : 각 구조물과 측정 지점을 기준으로 과거의 진단 결과를 저장하고 관리 
- 사용자 데이터 기반 재학습 : 실제 측정 데이터를 추가하여 건축물 특성을 반영하도록 AI모델 재학습 가능
- 진단 성능 확인 : 혼동행렬과 정확도, F1 score등을 통해 분류 성능을 검증 가능
- 센싱 디바이스 : 전자석을 이용해 구조물 내부 철근을 자화하고 자기응답 데이터를 측정하며, 자화 세기·방향 제어부터 센서 데이터 수집 및 Wi-Fi 기반 실시간 전송까지 일괄 수행

1-5.기대 효과 및 활용 분야 
기대 효과 :
- 예방 중심 관리 : 균열·손상 발생 이후가 아닌, 반복 측정을 통해 상태 변화를 조기에 확인
- 정량적·객관적 진단: B-H 특징값과 AI를 활용해 검사자의 주관 의존도를 낮추고 일관된 기준으로 상태 판단
- 진단 과정 자동화 및 비용 절감: 측정부터 분석·분류·시각화·이력 저장까지 통합하여 현장 운용 효율 향상
- 장기 상태 추적 및 예지보전 기반 확보: 누적 데이터를 통해 변화 추세를 관리하고 정밀점검·유지보수 우선순위 결정에 활용

활용 분야 :
- 교량·터널·철도 등 노후 SOC 인프라의 예방적 유지관리
- 건축물·공동주택 등 콘크리트 내부 철근 및 매립 강재 상태 점검
- 해양·플랜트·산업설비·에너지 구조물 등 반복하중을 받는 강재의 피로 관리
- 재난 이후 긴급 점검 및 현장용 비파괴 진단 보조 시스템

1-6.기술 스택 
- 프론트엔드 / GUI : Tkinter, Matplotlib
- 백엔드 / 데이터 처리 : Python, NumPy, SciPy, SQLite, CSV
- AI/ML : Scikit-learn, MLPClassifier, StandardScaler, Joblib
- 신호처리 / 물리 모델링 : FFT, Notch Filter, Moving RMS, Jiles-Atherton Model, RK4, Robust Least-Squares
- 임베디드 / 통신 : Raspberry Pi, Arduino UNO R4 WiFi, UDP, Wi-Fi
- 시뮬레이션 / 해석 : COMSOL Multiphysics
- 하드웨어 / 센서 : WSH135-XPAN2 Hall Sensor, ACS712 5A, MD13S, Yoke Electromagnet
- 개발 및 협업 : Visual Studio Code, Jupyter Notebook, Git, GitHub, MS Teams, Google Docs

2. 팀원 소개
## 👥 Team

<table>
  <tr>
    <td align="center">
      <img src="./한이음_사진모음/member1.png" width="110"><br>
      <b>팀원 1</b><br>
      역할
    </td>
    <td align="center">
      <img src="./한이음_사진모음/member2.png" width="110"><br>
      <b>팀원 2</b><br>
      역할
    </td>
    <td align="center">
      <img src="./한이음_사진모음/member3.png" width="110"><br>
      <b>팀원 3</b><br>
      역할
    </td>
    <td align="center">
      <img src="./한이음_사진모음/member4.png" width="110"><br>
      <b>팀원 4</b><br>
      역할
    </td>
    <td align="center">
      <img src="./한이음_사진모음/member5.png" width="110"><br>
      <b>멘토</b><br>
      Mentor
    </td>
  </tr>
</table>




3.시스템 구성도 








4.작품 소개영상 
[![영상 제목](유튜브 썸네일 URL)](https://www.youtube.com/watch?v=YVsy0BCYxd4)






5.핵심 소스코드 
python
- 1. pipeline.py : H 신호의 Moving RMS를 기반으로 실제 자화가 이루어진 구간을 검출하고, 불완전한 자화 구간을 제외하여 J-A fitting에 사용할 유효 cycle만 추출합니다.
{152-158}
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

{198-214}
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

-2. pipeline.py : 각 자화 cycle에서 독립적으로 J-A parameter를 추정한 뒤, 후보 parameter를 다른 모든 cycle에 다시 적용하는 Cross-Cycle Validation을 수행하여 특정 cycle에 과적합되지 않는 대표 parameter를 선택합니다.

{538-570}
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

{611-669}
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

            error = (B_pred - B_cycle)
            rmse = float(np.sqrt(np.mean(error ** 2)))
            B_range = float(np.ptp(B_cycle))

            if B_range > 1e-12:
                nrmse = (rmse / B_range * 100.0)
            else:
                nrmse = np.inf

            validation_nrmse.append(float(nrmse))

        candidate["validation_nrmse_list"] = validation_nrmse
        candidate["validation_nrmse_median"] = float(
            np.median(validation_nrmse)
        )

{671-676}
 selected_cycle = min(
        valid_cycles,
        key=lambda r: r["validation_nrmse_median"],)

-3. pipeline.py : 동일한 J-A parameter를 유지하면서 각 cycle의 초기 자화 상태 M_0만 별도로 보정하여 측정 cycle 간 초기 상태 차이를 보완합니다. 이를 통해 기준 데이터의 median NRMSE를 약 개선하였습니다.

{781-809}
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

{840-866}
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

{950-962}
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

    -4. ja_model : 큰 H 범위에서 발생할 수 있는 RK4 수치 폭주를 방지하기 위해 H 변화량을 내부 substep으로 나누어 J-A 미분방정식을 계산합니다.
    
{331-365}
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

{370-431}
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

-5. ja_fitting.py : J-A parameter들의 크기 차이를 보정하고, 순간적인 spike/outlier가 fitting 전체를 왜곡하지 않도록 Soft-L1 기반 Robust Least-Squares를 적용합니다.
{220-225}
B_target = B[start_index:]

    B_scale = np.ptp(B_target)

    if B_scale < 1e-12:
        B_scale = 1.0
{238-241}
 x_scale = np.maximum(
        np.abs(initial_params),
        1e-12,
    )
{248-272}
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
{292-318}
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
        
  -6. diagnosis_engine.py : 전체 진단 Pipeline을 실행한 뒤 복원된 Major Loop에서 Bmax, Area를 추출하고, 해당 특징값을 AI 진단 단계로 전달합니다.
  
 {58-89}
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

{91-107}
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

-7. mlp_model.py : 현재 프로토타입에서는 Bmax와 Area를 입력으로 사용하여 안정·주의·위험 상태를 분류합니다.
{40-83}
def diagnose(self, Bmax, Hmax, Area):
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

-8. sensor_udp_io.py : Raspberry Pi에서 Arduino에 측정 시작·종료 명령을 전송하고, 센싱 디바이스로부터 전송되는 데이터를 UDP로 실시간 수신합니다.
{52-67}
  def _send_command(self, command):
        if self.sock is None:
            self.connect()
        if command not in ("0", "1"):
            raise ValueError("Arduino command는 '0' 또는 '1'이어야 합니다.")
        self.sock.sendto(command.encode("ascii"), (self.arduino_ip, self.port))

    def start_measurement(self):
        self._send_command("1")

    def stop_measurement(self):
        if self.sock is not None:
            try:
                self._send_command("0")
            except OSError:
                pass
{75-109}
 def read_measurement(self, duration=DEFAULT_MEASUREMENT_SECONDS):
        if self.sock is None:
            self.connect()
        times, h_values, b_values = [], [], []
        self.start_measurement()
        first_t = None
        try:
            while True:
                try:
                    data, _addr = self.sock.recvfrom(256)
                except socket.timeout:
                    if first_t is None:
                        raise TimeoutError(
                            "Arduino UDP 패킷을 받지 못했습니다. Raspberry Pi가 "
                            "BH_Sensing Wi-Fi에 연결됐는지 확인하세요."
                        )
                    break
                pair = parse_packet(data)
                if pair is None:
                    continue
                now = time.monotonic()
                if first_t is None:
                    first_t = now
                t = now - first_t
                h, b = pair
                times.append(t)
                h_values.append(h)
                b_values.append(b)
                if t >= float(duration):
                    break
        finally:
            self.stop_measurement()
        if len(times) < 20:
            raise ValueError(f"수신 데이터가 너무 적습니다: {len(times)}개")
        return np.asarray(times), np.asarray(h_values), np.asarray(b_values)

- 9.mlp_retraining.py : COMSOL 기반 데이터를 이용해 초기 MLP를 학습하고, 사용자가 추가한 실제 측정 데이터를 누적하여 Active MLP를 다시 학습할 수 있습니다.
{28-55}
def build_mlp(random_state=0):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=35,
            random_state=random_state,
        ),
    )


def train_base_model(save=True):
    X, y = generate(per_class=BASE_PER_CLASS, seed=0)
    model = build_mlp(random_state=0)
    model.fit(X, y)

    X_test, y_test = generate(per_class=350, seed=991)
    acc = float(accuracy_score(y_test, model.predict(X_test)))

    if save:
        joblib.dump(model, BASE_MODEL_PATH)
        joblib.dump(model, ACTIVE_MODEL_PATH)
    return model, {"comsol_validation_accuracy": acc, "base_samples": int(len(y))}
{89-125}
ef retrain_active_model(training_records):
    """COMSOL 기본 데이터 + DB에 누적된 실측 라벨 특징으로 active MLP 재학습."""
    ensure_models()
    X_base, y_base = generate(per_class=BASE_PER_CLASS, seed=0)
    X_parts = [X_base]
    y_parts = [y_base]
    rng = np.random.default_rng(20260829)

    valid_real = 0
    for r in training_records:
        try:
            feature = [float(r["Bmax"]), float(r["Area"])]
            label_idx = int(r["label_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if label_idx not in (0, 1, 2) or not np.all(np.isfinite(feature)):
            continue
        Xr, yr = _augment_real_sample(feature, label_idx, rng)
        X_parts.append(Xr)
        y_parts.append(yr)
        valid_real += 1

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    model = build_mlp(random_state=0)
    model.fit(X, y)
    joblib.dump(model, ACTIVE_MODEL_PATH)

    X_test, y_test = generate(per_class=350, seed=991)
    acc = float(accuracy_score(y_test, model.predict(X_test)))
    return {
        "active_model_path": ACTIVE_MODEL_PATH,
        "comsol_validation_accuracy": acc,
        "base_samples": int(len(y_base)),
        "real_files": int(valid_real),
        "training_rows_total": int(len(y)),
    }
