# fft_analysis.py
from scipy.signal import find_peaks
import numpy as np


def estimate_sampling_frequency(time_data):
    """
    timestamp 배열로부터 sampling frequency를 계산한다.

    Parameters
    ----------
    time_data : array-like
        시간 데이터 [s]
        ex) [0.000, 0.001, 0.002, ...]

    Returns
    -------
    fs : float
        sampling frequency [Hz]
    """

    time_data = np.asarray(time_data, dtype=float)

    if len(time_data) < 2:
        raise ValueError("시간 데이터가 최소 2개 이상 필요합니다.")

    dt = np.diff(time_data)

    if np.any(dt <= 0):
        raise ValueError("시간 데이터는 반드시 증가하는 순서여야 합니다.")

    # UDP 등의 미세한 sampling jitter 때문에 평균보다 median 사용
    dt_median = np.median(dt)

    fs = 1.0 / dt_median

    return fs


def calculate_fft(signal, fs, remove_dc=True, use_hann=True):
    """
    1차원 센서 신호의 single-sided FFT를 계산한다.

    Parameters
    ----------
    signal : array-like
        시간순 센서 데이터 B(t) 또는 H(t)

    fs : float
        sampling frequency [Hz]

    remove_dc : bool
        평균값(DC offset) 제거 여부

    use_hann : bool
        Hann window 적용 여부

    Returns
    -------
    freq : ndarray
        주파수 축 [Hz]

    amplitude : ndarray
        각 주파수의 amplitude
    """

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal은 1차원 데이터여야 합니다.")

    if len(signal) < 4:
        raise ValueError("FFT를 수행하기에는 데이터 개수가 너무 적습니다.")

    if fs <= 0:
        raise ValueError("sampling frequency는 0보다 커야 합니다.")

    # NaN / inf 체크
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal 안에 NaN 또는 inf 값이 있습니다.")

    x = signal.copy()

    # DC offset 제거
    if remove_dc:
        x = x - np.mean(x)

    N = len(x)

    # Hann window
    if use_hann:
        window = np.hanning(N)
        x_windowed = x * window

        # Hann window amplitude 보정
        normalization = np.sum(window)
    else:
        x_windowed = x
        normalization = N

    # 양의 주파수 영역만 계산
    fft_result = np.fft.rfft(x_windowed)

    freq = np.fft.rfftfreq( N,  d=1.0 / fs )

    # single-sided amplitude spectrum
    amplitude = ( 2.0 * np.abs(fft_result) / normalization  )

    # DC 성분은 2배 하면 안 됨
    amplitude[0] /= 2.0

    # N이 짝수이면 Nyquist 성분도 2배 하면 안 됨
    if N % 2 == 0:
        amplitude[-1] /= 2.0

    return freq, amplitude


def find_dominant_frequencies(
    freq,
    amplitude,
    top_n=10,
    min_freq=0.1,
    max_freq=None,
    prominence_ratio=0.01,
    min_distance_hz=1.0,
):
    """
    FFT spectrum에서 실제 local peak를 검출한다.

    Parameters
    ----------
    freq : ndarray
        주파수 축 [Hz]

    amplitude : ndarray
        FFT amplitude

    top_n : int
        최대 반환 peak 개수

    min_freq : float
        최소 분석 주파수

    max_freq : float or None
        최대 분석 주파수

    prominence_ratio : float
        최대 amplitude 대비 최소 prominence 비율

    min_distance_hz : float
        서로 다른 peak로 인정할 최소 주파수 간격 [Hz]
    """

    freq = np.asarray(freq)
    amplitude = np.asarray(amplitude)

    mask = freq >= min_freq

    if max_freq is not None:
        mask &= freq <= max_freq

    f = freq[mask]
    a = amplitude[mask]

    if len(f) < 3:
        return []

    # 주파수 분해능
    df = f[1] - f[0]

    # 최소 peak 거리 → bin 개수
    distance_bins = max(
        1,
        int(min_distance_hz / df)
    )

    # 주변 noise floor 대비 얼마나 튀어나왔는지
    prominence_threshold = (
        np.max(a) * prominence_ratio
    )

    peaks, properties = find_peaks(
        a,
        prominence=prominence_threshold,
        distance=distance_bins
    )

    if len(peaks) == 0:
        return []

    # amplitude 큰 순서
    order = np.argsort(
        a[peaks]
    )[::-1]

    dominant = []

    for idx in order[:top_n]:

        peak_idx = peaks[idx]

        dominant.append(
            {
                "frequency": float(
                    f[peak_idx]
                ),
                "amplitude": float(
                    a[peak_idx]
                ),
                "prominence": float(
                    properties[
                        "prominences"
                    ][idx]
                ),
            }
        )

    return dominant

def analyze_signal(
    signal,
    fs,
    top_n=10,
    min_freq=0.1,
    max_freq=None,
):
    """
    센서 신호 하나에 대해 FFT 분석 수행.

    Returns
    -------
    result : dict

    {
        "fs": ...,
        "nyquist": ...,
        "resolution": ...,
        "freq": ...,
        "amplitude": ...,
        "dominant_frequencies": ...
    }
    """

    freq, amplitude = calculate_fft(
        signal=signal,
        fs=fs,
        remove_dc=True,
        use_hann=True,
    )

    resolution = fs / len(signal)

    dominant = find_dominant_frequencies(
        freq,
        amplitude,
        top_n=top_n,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    result = {
        "fs": float(fs),

        # FFT에서 분석 가능한 최고 주파수
        "nyquist_frequency": float(fs / 2.0),

        # 주파수 분해능
        "frequency_resolution": float(resolution),

        "freq": freq,
        "amplitude": amplitude,

        "dominant_frequencies": dominant,
    }

    return result


def analyze_bh_signals(
    B,
    H,
    fs,
    top_n=10,
    min_freq=0.1,
    max_freq=None,
):
    """
    B(t), H(t)를 각각 FFT 분석한다.

    Parameters
    ----------
    B : array-like
        B(t) 데이터

    H : array-like
        H(t) 데이터

    fs : float
        sampling frequency [Hz]

    Returns
    -------
    results : dict
    """

    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)

    if len(B) != len(H):
        raise ValueError(
            f"B와 H 데이터 길이가 다릅니다. "
            f"B={len(B)}, H={len(H)}"
        )

    B_result = analyze_signal(
        signal=B,
        fs=fs,
        top_n=top_n,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    H_result = analyze_signal(
        signal=H,
        fs=fs,
        top_n=top_n,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    return {
        "B": B_result,
        "H": H_result,
    }


def print_fft_summary(results):
    """
    analyze_bh_signals() 결과를 터미널에 출력한다.
    GUI용이 아니라 개발/분석용.
    """

    print("\n============================")
    print("       FFT ANALYSIS")
    print("============================")

    for name in ["B", "H"]:

        result = results[name]

        print(f"\n[{name} signal]")
        print(
            f"Sampling frequency : "
            f"{result['fs']:.3f} Hz"
        )
        print(
            f"Nyquist frequency  : "
            f"{result['nyquist_frequency']:.3f} Hz"
        )
        print(
            f"FFT resolution     : "
            f"{result['frequency_resolution']:.6f} Hz"
        )

        print("\nDominant frequencies:")

        for i, peak in enumerate(
            result["dominant_frequencies"],
            start=1,
        ):
            print(
                f"{i:2d}. "
                f"{peak['frequency']:10.3f} Hz "
                f"| amplitude = "
                f"{peak['amplitude']:.6g}"
            )







