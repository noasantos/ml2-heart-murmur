"""Pre-processing and feature extraction.

Pipeline for one recording (``recording_logmel``):

1. load the .wav (4000 Hz, mono, 16 bit in the whole public set);
2. band-pass, 25-800 Hz by default: heart sounds sit roughly at 25-200 Hz and
   murmurs, including high-pitched ones, reach a few hundred Hz; below 25 Hz
   is movement/handling noise and above 800 Hz mostly noise;
3. divide by the standard deviation, so the recording gain does not matter;
4. log-mel spectrogram with 64 bands covering the same band, 10 ms step
   (100 frames/s).

The whole recording is turned into one spectrogram and cached; fixed-length
windows are cut later (``src.data.WindowDataset``), so the window length can
change without recomputing features. The same spectrogram feeds both models:
the CNN reads a window as an image, the RNN as a sequence of vectors.

Variants for the pre-processing comparison:

- ``band``: every function takes the band-pass cut-offs; each band is cached
  in its own folder (``variant_name``), so variants live side by side.
- MFCCs: ``mfcc_from_logmel`` (20 coefficients + 20 deltas) is computed from
  the cached log-mel; ``WindowDataset(feature="mfcc")`` uses it.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.signal
import soundfile as sf

SR = 4000             # sampling rate of every recording in CirCor
BAND = (25.0, 800.0)  # default band-pass cut-offs in Hz
FILTER_ORDER = 4

N_MELS = 64
WIN_LENGTH = 200      # 50 ms analysis window
HOP_LENGTH = 40       # 10 ms step
FMIN, FMAX = BAND
FRAMES_PER_SECOND = SR / HOP_LENGTH  # 100

N_MFCC = 20           # MFCC coefficients; with deltas the feature has 2 * N_MFCC rows

FEATURES = ("logmel", "mfcc")


def _check_band(band) -> tuple[float, float]:
    lo, hi = float(band[0]), float(band[1])
    if not 0 < lo < hi < SR / 2:
        raise ValueError(f"band must satisfy 0 < low < high < {SR / 2:g} Hz, got {band}")
    return lo, hi


def variant_name(band=BAND) -> str:
    """Cache folder name of a pre-processing variant, e.g. ``band25-800``."""
    lo, hi = _check_band(band)
    return f"band{lo:g}-{hi:g}"


def n_features(feature: str = "logmel") -> int:
    """Rows of a feature window: 64 for log-mel, 40 for MFCC + deltas."""
    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}, got {feature!r}")
    return N_MELS if feature == "logmel" else 2 * N_MFCC


def load_audio(path: str | Path) -> np.ndarray:
    """Read a .wav as float32 in [-1, 1]. Raises if the sampling rate is not SR."""
    x, sr = sf.read(str(path), dtype="float32")
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr} Hz")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


@lru_cache(maxsize=None)
def _sos(band: tuple[float, float]) -> np.ndarray:
    return scipy.signal.butter(FILTER_ORDER, band, btype="bandpass", fs=SR, output="sos")


def bandpass(x: np.ndarray, band=BAND) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    return scipy.signal.sosfiltfilt(_sos(_check_band(band)), x).astype(np.float32)


def normalize_amplitude(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero mean, unit standard deviation."""
    x = x - x.mean()
    return (x / (x.std() + eps)).astype(np.float32)


def preprocess_signal(x: np.ndarray, band=BAND) -> np.ndarray:
    """Band-pass filter and amplitude normalization."""
    return normalize_amplitude(bandpass(x, band))


@lru_cache(maxsize=None)
def _mel_setup(band: tuple[float, float]) -> tuple[int, np.ndarray]:
    """FFT size and mel filterbank for a band.

    The analysis window stays 50 ms; the FFT is zero-padded (512, 1024, ...)
    until every one of the 64 mel filters covers at least one FFT bin. A
    narrow band (e.g. 25-400 Hz) needs finer bins than the default one.
    """
    import librosa

    lo, hi = band
    n_fft = 512
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=N_MELS, fmin=lo, fmax=hi)
        if (fb.max(axis=1) > 0).all():
            return n_fft, fb.astype(np.float32)
        n_fft *= 2
        if n_fft > 8192:
            raise ValueError(f"band {band} is too narrow for {N_MELS} mel bands")


def logmel(x: np.ndarray, band=BAND) -> np.ndarray:
    """Log-mel spectrogram (dB) of a pre-processed signal, shape (N_MELS, frames)."""
    import librosa

    n_fft, fb = _mel_setup(_check_band(band))
    stft = librosa.stft(x, n_fft=n_fft, hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH, window="hann", center=True)
    mel = fb @ (np.abs(stft) ** 2)
    return (10.0 * np.log10(np.maximum(mel, 1e-10))).astype(np.float32)


def recording_logmel(path: str | Path, band=BAND) -> np.ndarray:
    """Full pipeline for one .wav file: load, filter, normalize, log-mel."""
    return logmel(preprocess_signal(load_audio(path), band), band)


def mfcc_from_logmel(spec: np.ndarray, n_mfcc: int = N_MFCC, deltas: bool = True) -> np.ndarray:
    """MFCCs (DCT of the log-mel), optionally with first-order deltas.

    Works on a full spectrogram or a window, shape (N_MELS, frames) ->
    (n_mfcc or 2*n_mfcc, frames). Needs at least 9 frames for the deltas.
    """
    spec = np.asarray(spec, dtype=np.float32)
    mfcc = scipy.fft.dct(spec, type=2, axis=0, norm="ortho")[:n_mfcc]
    if not deltas:
        return mfcc.astype(np.float32)
    import librosa
    d = librosa.feature.delta(mfcc, width=9, axis=1)
    return np.concatenate([mfcc, d]).astype(np.float32)


def compute_feature(spec: np.ndarray, feature: str = "logmel") -> np.ndarray:
    """Feature matrix of a cached log-mel: the log-mel itself, or MFCC + deltas."""
    n_features(feature)  # validates the name
    spec = np.asarray(spec, dtype=np.float32)
    return spec if feature == "logmel" else mfcc_from_logmel(spec)
