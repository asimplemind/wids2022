"""test_audio.py unittests for audio.py"""
import os
import unittest
from pathlib import Path
import soundfile as sf

from acoustic_ml import audio


class TestAudioConversion(unittest.TestCase):
    """AudioConversion Unittest Class"""

    def test_load_audio_file(self):
        """test load_audio_file function"""
        base_path = "tests"
        data_path = "test_data"
        out_path = "out"

        if not os.path.exists(os.path.join(base_path, out_path)):
            Path(os.path.join(base_path, out_path)).mkdir(parents=True)

        file1 = Path(os.path.join(base_path, data_path, "sample_flac.flac"))
        y, _ = audio.load_audio_file(file1, sr=44100, duration=5)
        sf.write(os.path.join(base_path, out_path, "tmp_flac.wav"), y, samplerate=44100)
        self.assertEqual((220500,), y.shape)

        file2 = Path(os.path.join(base_path, data_path, "sample_wav.wav"))
        y, _ = audio.load_audio_file(file2, sr=44100, duration=5)
        sf.write(os.path.join(base_path, out_path, "tmp_wav.wav"), y, samplerate=44100)
        self.assertEqual((220500,), y.shape)


if __name__ == '__main__':
    unittest.main()
