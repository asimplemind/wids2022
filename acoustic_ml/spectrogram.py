"""spectrogram.py
Create, display, and save different types of spectrogram for processing
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display


def create_spectrogram_librosa(y: np.ndarray, cfg: dict) -> np.ndarray:
    """
    create_spectrogram -- takes in a time series and output the power spectrum in decibel

    Arguments:
        y <numpy.ndarray>: time series
        cfg <dict>: configuration parameters in a dictionary
    Return:
        <numpy.ndarray>: power spectrum in decibel
    """
    spectrum = librosa.stft(y=y,
                            n_fft=cfg["spectrogram"]["nfft"],
                            hop_length=cfg["spectrogram"]["nfft"]//2,
                            window=cfg["spectrogram"]["window"])
    return librosa.core.amplitude_to_db(np.abs(spectrum))


def create_spectrogram_scipy(y: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    create_spectrogram -- takes in a time series and output the power spectrum in decibel

    Arguments:
        y <numpy.ndarray>: time series
        cfg <dict>: configuration parameters in a dictionary
    Return:
        f <numpy.ndarray>: sample frequencies
        t <numpy.ndarray>: segment times
        spectrum <np.ndarray>: power spectrum in decibel
    """
    freq, time, spectrum = signal.spectrogram(y,
                                              fs=cfg['audio']['sampling_rate'],
                                              scaling='density',
                                              window=cfg['spectrogram']['window'],
                                              nfft=cfg['spectrogram']['nfft'])
    return freq, time, 10*np.log10(spectrum)


def create_melspectrogram(y: np.ndarray, cfg: dict):
    """
    Calculate the spectrogram in mel scale that is sensitive to human's range

    Arguments:
        y <numpy.ndarray>: time series data
        cfg <dict>: dictionary with configuration parameters
    Returns:
        <numpy.ndarray>: mel-spectrogram
    """

    return librosa.feature.melspectrogram(y,
                                          sr=cfg['audio']['sampling_rate'],
                                          fmin=cfg['melspec']['fmin'],
                                          fmax=cfg['melspec']['fmax'],
                                          n_mels=cfg['melspec']['n_mels'])


def create_pcen(y: np.ndarray, cfg: dict):
    """
    Compute the per-channel-energy normalization

    Arguments:
        y <numpy.ndarray>: input series
        cfg <dict>: dictionary structure with configuration parameters
    Returns:
        <numpy.ndarray> per-channel-energy normalized data
    """

    return librosa.core.pcen(y * (2**31),
                             sr=cfg['audio']['sampling_rate'],
                             fmax=cfg['audio']['sampling_rate'] // 2,
                             gain=cfg['pcen']['gain'],
                             bias=cfg['pcen']['bias'],
                             power=cfg['pcen']['power'],
                             hop_length=cfg['spectrogram']['nfft']//2,
                             time_constant=cfg['pcen']['time_constant'],  # for IIR filtering
                             eps=np.finfo(float).eps)


def display_spectrogram(spec: np.ndarray) -> None:
    """
    librosa allows a quick visualization of spectrogram, for more fine tuning of plot,
    Use matplotlib.pyplot pcolormesh

    Arguments:
        spec <numpy.ndarray>: spectrogram to be plotted
    Returns: None
    """
    fig, ax = plt.subplots(figsize=(3, 2))
    img = librosa.display.specshow(spec, y_axis='hz', x_axis='time')
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


def save_spectrogram(spec: np.ndarray, freq: np.ndarray, time: np.ndarray, spectrogram_path: Path) -> None:
    """
    librosa allows a quick visualization of spectrogram, for more fine tuning of plot,
    Use matplotlib.pyplot pcolormesh

    Arguments:
        spec <numpy.ndarray>:  spectrogram to be saved
        freq <numpy.ndarray>: frequency axis marks
        time <numpy.ndarray>: time axis marks
        spectrogram_path <Path>: path to save the spectrogram
    Returns: None
    """
    plt.subplots(figsize=(3, 2))
    plt.pcolormesh(time, freq, spec, vmin=spec.min()+10, vmax=spec.max(), cmap='YlGnBu_r')
    plt.axis('off')
    plt.tight_layout()
    print(spectrogram_path)
    plt.savefig(spectrogram_path, dpi=100)
    plt.close()
