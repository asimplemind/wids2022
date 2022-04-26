"""This data loader file is created specifically for WHOI's dolphin dataset"""

import os
import sys
from pathlib import Path
from typing import Tuple
import logging
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from acoustic_ml import audio, spectrogram

logger = logging.getLogger(__name__)
logger.info("In data loader")

plt.rcParams['figure.figsize'] = [2, 1]


class DataLoader:
    """DataLoader

    Functions for general splitting and creating spectrogram
    create_train_test_dataset
    create_spectrograms
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Set up paths
        self.data_path = Path(cfg['paths']['data_path'])
        self.wav_path = self.data_path / cfg['paths']['wav_path']
        self.outputs_path = Path(cfg['paths']['output_path'])
        self.spec_path = self.outputs_path / 'spectrogram'

    def create_train_test_dataset(self):
        """This step split the data available to train and validation folders

        This function do the following steps:
        1. Gather filepath from the positive (pos) and negative (neg) folders with jpg or png
        2. Obtain a balanced data set using the smaller set between the pos or neg sets
        3. Split data to train and validation data (performed separately for pos/neg to ensure good
        mix of both pos and neg data in train/val, before merging pos/neg data for train/val sets)
        4. Make the data into dataframe format to use with tf.keras.preprocessing.image.ImageDataGenerator
        """

        # --------------------------------------------------------------------------------------------------
        # 1. Gather filepath from the folders with jpg or png
        #
        # Create this directory structure
        # outputs
        # |
        # |--train
        #     |-- Dolphin1 (class)
        #     |-- Dolphin2
        #     |-- Dolphin3
        #     |-- Dolphin4
        # |--val
        #     |-- Dolphin1
        #     |-- Dolphin2
        #     |-- Dolphin3
        #     |-- Dolphin4
        # |--test
        #     |-- Dolphin1
        #     |-- Dolphin2
        #     |-- Dolphin3
        #     |-- Dolphin4
        #
        # Make the data into dataframe format to use with tf.keras.preprocessing.image.ImageDataGenerator
        # --------------------------------------------------------------------------------------------------

        train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)
        for root, _, files in os.walk(self.spec_path):
            for file in files:
                if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
                    split_path = root.split(os.sep)
                    if split_path[-2] == 'train':
                        train['filename'].append(os.path.join(root, file))
                        train['class'].append(split_path[-1])
                    elif root.split(os.sep)[-2] == 'val':
                        val['filename'].append(os.path.join(root, file))
                        val['class'].append(split_path[-1])
                    elif root.split(os.sep)[-2] == 'test':
                        test['filename'].append(os.path.join(root, file))
                        test['class'].append(split_path[-1])

        print("\nSummary of create_train_test_dataset")
        print(f"Number of files available {len(train)} train, {len(val)} val, and {len(test)} test")

        # --------------------------------------------------------------------------------------------------
        # 2. Convert data into dataframe format to use with tf.keras.preprocessing.image.ImageDataGenerator
        # --------------------------------------------------------------------------------------------------
        df_train = pd.DataFrame(train)
        df_train = df_train.sample(frac=1)
        df_val = pd.DataFrame(val)
        df_val = df_val.sample(frac=1)
        df_test = pd.DataFrame(test)
        df_test = df_test.sample(frac=1)

        # save unique class labels for training
        unique_class_labels = np.sort(list(set(df_train['class'].to_list())))
        print(f"Unique classes found in data are {unique_class_labels}\n")

        return df_train, df_val, df_test, unique_class_labels

    def create_spectrograms(self):
        """Simple function to look at directories of positive or negative examples and convert to spectrograms
        Good if you don't need to split long audio recordings up using metadata.
        """
        for root, _, files in os.walk(self.wav_path):
            for file in files:
                if file.lower().endswith('.flac') or file.lower().endswith('.wav'):
                    data, _ = audio.load_audio_file(Path(os.path.join(root, file)))
                    freq, time, spec = spectrogram.create_spectrogram_scipy(data, self.cfg)

                    # some post-processing on the spectrogram learned from collaboration with WHOI
                    median = np.percentile(spec, q=50)
                    spec[spec < median] = median

                    # pad or crop images as needed
                    time, spec = self.pad_images(self.cfg, time, spec)

                    split_type = self.get_train_val_test_type()
                    label_path = os.path.join(self.spec_path, split_type, os.path.split(root)[-1])
                    if not os.path.exists(label_path):
                        Path(label_path).mkdir(parents=True)

                    file_name = os.path.join(label_path, os.path.splitext(file)[0] + '.jpg')
                    spectrogram.save_spectrogram(spec=spec, freq=freq, time=time, spectrogram_path=Path(file_name))

        # Obtain image shape for checking dimensions
        if self.cfg['spectrogram']['image_shape'] is None:
            self.cfg['spectrogram']['image_shape'] = self.get_image_shape(Path(file_name))


    @staticmethod
    def get_image_shape(image_path: Path) -> Tuple:
        """obtain image shape and store as a configuration parameter

        Arguments:
            image_path <Path>: path to an image of interest
        Returns:
            <Tuple>: the image shape
        """
        print(f'Obtaining image_shape for an image sample - {image_path}')
        img = cv2.imread(str(image_path))
        return img.shape


    @staticmethod
    def pad_images(cfg: dict, time: np.ndarray, spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """pad_images

        Arguments:
            cfg <dict>: configuration file containing all configurable parameters
            time <numpy.ndarray>: time array to be padded if necessary
            spec <numpy.ndarray>: spectrogram to be padded if necessary
        Returns:
            time <numpy.ndarray>: padded time array
            spec <numpy.ndarray>: padded spectrogram
        """
        num_sample = int(np.ceil(cfg['spectrogram']['max_length_sec'] / time[0]))

        # need padding in the x-axis
        padding_required = num_sample - spec.shape[1]
        if padding_required > 0:
            left_padding = padding_required // 2
            right_padding = padding_required - left_padding
            pad_value = np.float64(spec.min())
            spec = cv2.copyMakeBorder(spec, 0, 0, left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
            time = np.linspace(time[0], cfg['spectrogram']['max_length_sec'], num_sample)
            # print(f'After padding, spec size is {spec.shape}')

        if padding_required < 0:
            # print(f'Need to crop image, size is {spec.shape}')
            left_padding = np.abs(padding_required) // 2
            right_padding = np.abs(padding_required) - left_padding

            spec = spec[:, left_padding:(spec.shape[1]-right_padding)]
            time = np.linspace(time[0], cfg['spectrogram']['max_length_sec'], num_sample)
            # print(f'After cropping, spec size is {spec.shape}')

        return time, spec

    @staticmethod
    def get_train_val_test_type(split_ratio: list = None) -> str:
        """get_train_val_test_type, a pretty long function name, this returns a string that indicate what the current
        sample should be put into -- train, validation, or test set.

        Arguments:
            split_ratio <list>: ratio for how train/val/test should be allocated percentage wise [test, val, test]
        Returns:
            <str>: in the form of 'test', 'val', or 'test
        """
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]
        if sum(split_ratio) != 1:
            print(f'split ratio is {split_ratio}, must sum up to 1')
            sys.exit()

        rand = random.random()
        if rand < split_ratio[0]:
            return 'train'
        if rand < sum(split_ratio[:2]):
            return 'val'
        return 'test'
