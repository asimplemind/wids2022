"""test_spectrogram.py unittests for spectrogram.py"""
import os
import unittest
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns

from acoustic_ml import audio
from acoustic_ml import spectrogram as spec

sns.set()


class TestSpectrogram(unittest.TestCase):
    """Spectrogram Unittest Class"""

    @staticmethod
    def test_create_spectrogram():
        """
        Comparing the two different spectrograms using librosa and scipy.
        The reference max is different, but the dynamic range is set to about the same
        """
        with open("config.json", "r",  encoding='UTF-8') as f:
            cfg = json.load(f)
        cfg['audio']['sampling_rate'] = 44100
        cfg['audio']['duration'] = 5

        base_path = "tests"
        data_path = "test_data"
        file = Path(os.path.join(base_path, data_path, "sample_wav.wav"))
        y, _ = audio.load_audio_file(file,
                                     sr=cfg['audio']['sampling_rate'],
                                     duration=cfg['audio']['duration'])

        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        spectrum_dB = spec.create_spectrogram_librosa(y, cfg)
        librosa.display.specshow(spectrum_dB, sr=cfg['audio']['sampling_rate'],
                                 y_axis='hz', x_axis='time', cmap='YlGnBu_r', ax=ax[0])
        ax[0].set(title='Spectrogram with Librosa STFT and Specshow')
        ax[0].set(ylabel='Frequency (KHz)')
        ax[0].set_yticks(np.array(np.linspace(0, cfg['audio']['sampling_rate']//2, 5)),
                         np.array(np.linspace(0, cfg['audio']['sampling_rate']//2000, 5), dtype=int))
        ax[0].label_outer()

        dynamic_range = 65  # in dB
        f, t, spectrum_dB = spec.create_spectrogram_scipy(y, cfg)
        plt.pcolormesh(t, f, spectrum_dB,
                       vmin=spectrum_dB.max()-dynamic_range, vmax=spectrum_dB.max(),
                       cmap='YlGnBu_r')
        ax[1].set(title='Spectrogram with Scipy Signal and pcolormesh')
        ax[1].set(xlabel='Time (second)')
        ax[1].set(ylabel='Frequency (KHz)')
        ax[1].set_xticks(np.array(np.linspace(t[0], t[-1], 6)), np.array(np.linspace(0, 5, 6), dtype=int))
        ax[1].set_yticks(np.array(np.linspace(t[0], f[-1], 5)),
                         np.array(np.linspace(0, cfg['audio']['sampling_rate']//2000, 5), dtype=int))

        #fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.show()

    @staticmethod
    def test_create_melspectrogram():
        """
        Test spectrogram's create_melspectrogram function
        """
        with open("config.json", "r",  encoding='UTF-8') as f:
            cfg = json.load(f)
        cfg['audio']['sampling_rate'] = 44100
        cfg['audio']['duration'] = 5

        base_path = "tests"
        data_path = "test_data"
        file = Path(os.path.join(base_path, data_path, "sample_wav.wav"))
        y, sr = audio.load_audio_file(file,
                                      sr=cfg['audio']['sampling_rate'],
                                      duration=cfg['audio']['duration'])

        fig, ax = plt.subplots()
        melspec = spec.create_melspectrogram(y, cfg)
        melspec_dB = librosa.power_to_db(melspec, ref=np.max)
        img = librosa.display.specshow(melspec_dB, x_axis='time', y_axis='mel',
                                       sr=sr, fmax=cfg['melspec']['fmax'], ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_yticks(np.array(np.linspace(cfg['melspec']['fmin'], cfg['melspec']['fmax'], 5)),
                      np.array(np.linspace(cfg['melspec']['fmin'], cfg['melspec']['fmax']//1000, 5), dtype=int))
        ax.set(title='Mel-frequency spectrogram')
        ax.set(ylabel='Mel-Frequency (KHz)')


if __name__ == '__main__':
    unittest.main()
