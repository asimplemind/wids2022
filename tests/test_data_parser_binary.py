import os
import sys
from pathlib import Path
import unittest
import json

sys.path.append('.')
from acoustic_ml import data_parser_binary


class TestDataParser(unittest.TestCase):

    def test_load_metadata(self):
        with open("tests/test_config.json", "r", encoding='UTF-8') as f:
            cfg = json.load(f)

        data_parser_obj = data_parser_binary.DataParserBinary(cfg)
        df = data_parser_obj.load_metadata(Path(os.path.join('tests', 'test_data', 'train.tsv')))
        self.assertIn('wav_filename', df.columns)

    def test_get_positive_samples(self):
        with open("tests/test_config.json", "r", encoding='UTF-8') as f:
            cfg = json.load(f)

        obj = data_parser_binary.DataParserBinary(cfg)
        obj.df = obj.load_metadata(Path(os.path.join('tests', 'test_data', 'train.tsv')))
        obj.start_time_sec = obj.df['start_time_s'].tolist()
        obj.duration_sec = obj.df['duration_s'].tolist()
        obj.file_names = obj.df['wav_filename'].tolist()

        obj.data_path = Path(os.path.join('tests', 'test_data'))
        obj.wav_path = obj.data_path
        obj.outputs_path = obj.data_path
        obj.pos_spec_path = Path(obj.data_path / 'pos_spec')
        if not os.path.exists(obj.pos_spec_path):
            obj.pos_spec_path.mkdir(parents=True)

        # call the function under test
        obj.get_positive_samples()

        num_files = len(os.listdir(obj.pos_spec_path))
        self.assertEqual(11, num_files)

    def test_get_negative_samples(self):
        with open("tests/test_config.json", "r", encoding='UTF-8') as f:
            cfg = json.load(f)

        obj = data_parser_binary.DataParserBinary(cfg)
        obj.df = obj.load_metadata(Path(os.path.join('tests', 'test_data', 'train.tsv')))
        obj.start_time_sec = obj.df['start_time_s'].tolist()
        obj.duration_sec = obj.df['duration_s'].tolist()
        obj.file_names = obj.df['wav_filename'].tolist()

        obj.data_path = Path(os.path.join('tests', 'test_data'))
        obj.wav_path = obj.data_path
        obj.outputs_path = obj.data_path
        obj.neg_spec_path = Path(obj.data_path / 'neg_spec')
        if not os.path.exists(obj.neg_spec_path):
            obj.neg_spec_path.mkdir(parents=True)

        # call the function under test
        obj.get_negative_samples()

        num_files = len(os.listdir(obj.neg_spec_path))
        self.assertEqual(13, num_files)


if __name__ == '__main__':
    unittest.main()
