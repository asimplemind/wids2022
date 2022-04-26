"""Parse long audio file given start_time"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from acoustic_ml import audio, spectrogram


class DataParserBinary:
    """DataParserBinary

    This class is tailored to processing long acoustic file with start_time, modify as needed to match metadata.
    If your data is already in yes or no folder format, you won't need most of this code

    Functions for parsing long audio files with start timestamp in seconds
    get_positive_samples (coarse extraction, can be further breakdown to classes for classification)
    get_negative_samples
    load_metadata

    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Set up paths
        self.data_path = Path(cfg['paths']['data_path'])
        self.wav_path = self.data_path / cfg['paths']['wav_path']
        self.outputs_path = Path(cfg['paths']['output_path'])
        self.pos_spec_path = self.outputs_path / cfg['paths']['pos_spec_path']
        self.neg_spec_path = self.outputs_path / cfg['paths']['neg_spec_path']

        self.extract_new_data = cfg['data']['extract_new_data']
        if self.extract_new_data:
            # Read in metadata that comes with the wav files (i.e. train.csv)
            self.wav_path = os.path.dirname(cfg['paths']['metadata']) / Path(cfg['paths']['wav_path'])
            metadata = cfg['paths']['metadata']
            self.df = self.load_metadata(metadata)
            self.start_time_sec = self.df['start_time_s'].tolist()
            self.duration_sec = self.df['duration_s'].tolist()
            self.file_names = self.df['wav_filename'].tolist()

            # Create output paths for images if not already exists
            if not os.path.exists(self.pos_spec_path):
                self.pos_spec_path.mkdir(parents=True)
            if not os.path.exists(self.neg_spec_path):
                self.neg_spec_path.mkdir(parents=True)

            self.get_positive_samples()
            self.get_negative_samples()

    def get_positive_samples(self):
        """Obtain positive samples from data"""
        # Loop through files, give it a unique name since each file might have multiple segments
        for i, file in enumerate(self.file_names):
            current_sec = self.start_time_sec[i]

            # each detected segment might span longer than specified duration,
            # try to obtain as many positive examples as possible
            while current_sec < self.start_time_sec[i] + self.duration_sec[i] - 0.5:  # give some buffer
                y, _ = audio.load_audio_file(self.wav_path / file,
                                             sr=self.cfg['audio']['sampling_rate'],
                                             offset=max(0, current_sec - 0.25),
                                             duration=self.cfg['audio']['duration'])
                freq, time, spec = spectrogram.create_spectrogram_scipy(y, self.cfg)

                # some post-processing on the spectrogram learned from collaboration with WHOI
                _median = np.percentile(spec, q=self.cfg['spectrogram']['percentile_threshold'])
                spec[spec < _median] = _median

                spectrogram.save_spectrogram(spec, freq, time,
                                             Path(os.path.join(self.pos_spec_path, os.path.splitext(file)[0]
                                                               + '-yes-' + str(round(current_sec, 2)) + '.jpg')))

                # update time
                current_sec = current_sec + self.cfg['audio']['duration']

    def get_negative_samples(self):
        """get_negative_samples.

        This code looks in between annotated data time series and
        extract everything in between the labeled positive time segments.
        Requires some visual checks after.
        """
        print(self.df.head(5))
        new_df = self.df.sort_values(by=['wav_filename', 'start_time_s']).copy().reset_index()

        # previous state
        last_file = new_df.wav_filename[0]  # to track which file we last looked at

        # set state for audio segmentation
        curr_time = 10  # current time point for segmenting (start at 10 sec to give a bit of buffer)
        negative_samples = []  # to keep a list of start time for the potential negative examples

        for file, start, duration in zip(new_df.wav_filename, new_df.start_time_s, new_df.duration_s):

            # set last time point (use current for the first loop)
            last_start = start
            last_duration = duration

            if last_file == file:  # if we are still processing the same file

                # while there are enough time series to extract samples of length duration_sec, continues
                while curr_time < start - self.cfg['spectrogram']['max_length_sec']:
                    negative_samples.append([file, curr_time])
                    curr_time += 2.5
                curr_time = last_start + last_duration  # new starting point for the next segment

            else:
                last_file = file
                curr_time = 10

        # save the negative sample plots
        for i, _ in enumerate(negative_samples):
            # load audio file and create spectrogram
            y_data, _ = audio.load_audio_file(self.wav_path / negative_samples[i][0],
                                              sr=self.cfg['audio']['sampling_rate'],
                                              offset=max(0, negative_samples[i][1] - 0.25),
                                              duration=self.cfg['spectrogram']['max_length_sec'])
            freq, time, spec = spectrogram.create_spectrogram_scipy(y_data, self.cfg)

            # add some post-processing on the spectrogram for better dynamic range
            _median = np.percentile(spec, q=self.cfg['spectrogram']['percentile_threshold'])
            spec[spec < _median] = _median

            # save spectrogram
            spectrogram.save_spectrogram(spec, freq, time,
                                         Path(os.path.join(self.neg_spec_path,
                                                           'no-' + str(round(negative_samples[i][1], 2)) + '.jpg')))

        return new_df, negative_samples

    @staticmethod
    def load_metadata(metadata_file: Path):
        """load metadata provided with the dataset

        Arguments:
            metadata_file <Path>: path include file name to the metadata file
        Returns:
            df <pandas.DataFrame>: a dataframe containing the metadata
        """
        df = pd.read_csv(str(metadata_file), delimiter='\t')
        print(df.describe())
        return df
