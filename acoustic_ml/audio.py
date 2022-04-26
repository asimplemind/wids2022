"""This file contains various audio file format conversion code and code for reading in different
audio file formats
"""

import os
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa
import ffmpy


def load_audio_file(file: Path, sr: int = None, offset: float = 0.0, duration: float = None) -> Tuple:
    """Use the native sampling frequency to start, unless you know what sampling rate you want to process
    the files with

    Arguments:
        file <Path>: file data path
        sampling_rate <int>: sampling rate of the file, default to None for native rate
        offset <float>: number of seconds to offset at the start of an audio file
        duration <float>: number of seconds to extract when reading the audio file
    Returns:
        data, sampling_rate <Tuple>: time series of the audio file, sampling rate used
    """

    filename = os.path.split(file)[-1]

    if filename.lower().endswith(".flac"):
        # Free Lossless Audio Codec (flac), an open source audio compression format
        data_flac, sr = sf.read(file)

        # check number of channels
        if len(data_flac.shape) > 1:  # if there's more than 1 dimension
            data = np.sum(data_flac, axis=-1) / data_flac.shape[-1]
        else:
            data = data_flac

    elif filename.lower().endswith(".wav") or filename.lower().endswith("m4a"):
        # Raw audio format by Microsoft and IBM, lossless
        # set duration to x seconds if you don't want to read in the full file
        # set sr=None if you want to obtain the file's original sampling rate

        # by default, librosa converts audio to mono
        if not duration:
            data, sr = librosa.load(file, sr=sr, offset=offset)
        else:
            data, sr = librosa.load(file, sr=sr, offset=offset, duration=duration)

    elif filename.lower().endswith(".ogg"):
        # Ogg Vorbis Compressed Audio file, include metadata
        # pip install PyOgg (pypi.org/project/PyOgg/)
        raise NotImplementedError

    if sr and duration:
        # clip data to ensure they are of equal size based on the sampling rate
        data = data[:sr * duration]

    return data, sr


def m4a_to_wav(audio_files, delete: bool = False) -> None:
    """
    Takes in .m4a files, convert to .wav and saved in the same directory as the m4a files located.

    Arguments:
        audio_files <dict>: dictionary of {audio_audio.m4a's id: audio_audio.m4a's filename}
        delete <bool>: flag to delete file m4a or not
    Returns: None
    """
    for filename in audio_files.values():
        if filename.lower().endswith(".m4a") == ".DS_Store":
            continue
        try:
            ff = ffmpy.FFmpeg(
                inputs={filename: None},
                outputs={filename[0:-4] + ".wav": None})
            if not os.path.isfile(filename[0:-4] + ".wav"):
                ff.run()
                if delete:
                    print("deleting", filename)
                    os.remove(filename)
        except OSError as e:
            print(e)
