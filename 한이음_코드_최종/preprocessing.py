# preprocessing.py

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


# ============================================================
# 공통 입력 검증
# ============================================================

def _validate_signal(signal, fs):
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("signal은 1차원 배열이어야 합니다.")

    if len(signal) < 10:
        raise ValueError("필터링하기에는 데이터가 너무 짧습니다.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal에 NaN 또는 inf가 포함되어 있습니다.")

    if fs <= 0:
        raise ValueError("sampling frequency fs는 0보다 커야 합니다.")

    return signal


# ============================================================
# Low-pass filter
# ============================================================

def lowpass_filter(
    signal,
    fs,
    cutoff,
    order=4,
):
    """
    Butterworth Low-pass Filter

    Parameters
    ----------
    signal : array-like
        입력 신호 B(t) 또는 H(t)

    fs : float
        sampling frequency [Hz]

    cutoff : float
        cutoff frequency [Hz]

    order : int
        Butterworth filter order

    Returns
    -------
    filtered : ndarray
        필터링된 신호
    """

    signal = _validate_signal(signal, fs)

    nyquist = fs / 2

    if cutoff <= 0:
        raise ValueError("cutoff는 0보다 커야 합니다.")

    if cutoff >= nyquist:
        raise ValueError(
            f"cutoff={cutoff} Hz는 "
            f"Nyquist frequency={nyquist} Hz보다 작아야 합니다."
        )

    sos = butter(
        N=order,
        Wn=cutoff,
        btype="lowpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        signal,
    )

    return filtered


# ============================================================
# Band-pass filter
# ============================================================

def bandpass_filter(
    signal,
    fs,
    lowcut,
    highcut,
    order=4,
):
    """
    Butterworth Band-pass Filter
    """

    signal = _validate_signal(signal, fs)

    nyquist = fs / 2

    if lowcut <= 0:
        raise ValueError("lowcut은 0보다 커야 합니다.")

    if highcut >= nyquist:
        raise ValueError(
            "highcut은 Nyquist frequency보다 작아야 합니다."
        )

    if lowcut >= highcut:
        raise ValueError(
            "lowcut은 highcut보다 작아야 합니다."
        )

    sos = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        signal,
    )

    return filtered


# ============================================================
# Notch filter
# ============================================================

def notch_filter(signal,fs, notch_freq, q=30.0,):
    """
    특정 주파수만 제거하는 Notch Filter.

    예:
        60 Hz 전원 노이즈 제거
    """

    signal = _validate_signal(signal, fs)

    nyquist = fs / 2

    if notch_freq <= 0:
        raise ValueError("notch_freq는 0보다 커야 합니다."  )

    if notch_freq >= nyquist:
        raise ValueError("notch_freq는 Nyquist frequency보다 작아야 합니다."  )

    if q <= 0:
        raise ValueError( "Q factor는 0보다 커야 합니다.")

    b, a = iirnotch( w0=notch_freq, Q=q, fs=fs,)
    filtered = filtfilt( b, a, signal, )
    return filtered


# ============================================================
# B / H 동시 LPF
# ============================================================

def lowpass_bh(
    B,
    H,
    fs,
    cutoff,
    order=4,
):
    """
    B(t), H(t)에 동일한 LPF를 적용한다.
    """

    B = np.asarray(B, dtype=float)
    H = np.asarray(H, dtype=float)

    if len(B) != len(H):
        raise ValueError(
            f"B/H 길이가 다릅니다. "
            f"B={len(B)}, H={len(H)}"
        )

    B_filtered = lowpass_filter(
        signal=B,
        fs=fs,
        cutoff=cutoff,
        order=order,
    )

    H_filtered = lowpass_filter(
        signal=H,
        fs=fs,
        cutoff=cutoff,
        order=order,
    )

    return B_filtered, H_filtered










