"""This data loader file is created specifically for Orcasound's acoustic sandbox data
Check out their official site for their implementations! :)
"""

import os
from pathlib import Path
import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from acoustic_ml import audio, spectrogram

logger = logging.getLogger(__name__)
logger.info("In data loader")

plt.rcParams['figure.figsize'] = [6, 3]


class DataLoaderBinary:
    """DataLoaderBinary

    Functions for general splitting and creating spectrograms
    create_train_test_dataset
    create_spectrograms
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Set up paths
        self.data_path = Path(cfg['paths']['data_path'])
        self.wav_path = self.data_path / cfg['paths']['wav_path']
        self.outputs_path = Path(cfg['paths']['output_path'])
        self.pos_spec_path = self.outputs_path / cfg['paths']['pos_spec_path']
        self.neg_spec_path = self.outputs_path / cfg['paths']['neg_spec_path']

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
        # 1. Gather filepath from the positive and negative folders with jpg or png
        # --------------------------------------------------------------------------------------------------
        # This is way more lines than needs to be, there are fancy calls that will do this efficiently
        positives_full = [fp for fp in os.listdir(self.pos_spec_path)
                          if fp.lower().endswith('.png') or fp.lower().endswith('.jpg')]
        negatives_full = [fp for fp in os.listdir(self.neg_spec_path)
                          if fp.lower().endswith('.png') or fp.lower().endswith('.jpg')]

        print("Summary of create_train_test_dataset")
        print(f"\nNumber of files available {len(positives_full)} positives and {len(negatives_full)} negatives")

        # --------------------------------------------------------------------------------------------------
        # 2. Obtain a balanced data set using the smaller set between the pos or neg sets
        #    random sampling without replacement
        # --------------------------------------------------------------------------------------------------
        if len(positives_full) > len(negatives_full):
            positives = random.sample(positives_full, k=len(negatives_full))
            negatives = negatives_full
        else:
            negatives = random.sample(negatives_full, k=len(positives_full))
            positives = positives_full

        print(f"After random sampling, number of files available {len(positives)} positives "
              f"and {len(negatives)} negatives", end='\n')

        # --------------------------------------------------------------------------------------------------
        # 3. Split data to train and validation data
        # --------------------------------------------------------------------------------------------------
        positives = [os.path.join(self.pos_spec_path, fp) for fp in positives]
        negatives = [os.path.join(self.neg_spec_path, fp) for fp in negatives]
        print(negatives[:5], positives[:2])
        X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(positives,
                                                                            ['1' for _ in range(len(positives))],
                                                                            test_size=0.20, random_state=1)
        X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(negatives,
                                                                            np.zeros(len(negatives), dtype=int),
                                                                            test_size=0.20, random_state=1)

        print("Summary of split")
        print(f"X_train_pos: {len(X_train_pos)}, y_train_pos: {len(y_train_pos)}")
        print(f"X_train_neg: {len(X_train_neg)}, y_train_neg: {len(y_train_neg)}")
        print(f"X_test_pos: {len(X_test_pos)}, y_test_pos: {len(y_test_pos)}")
        print(f"X_test_neg: {len(X_test_neg)}, y_test_neg: {len(y_test_neg)}")

        # merge positive and negative samples together
        X_train = X_train_pos + X_train_neg
        y_train = list(y_train_pos) + list(y_train_neg)
        X_test = X_test_pos + X_test_neg
        y_test = list(y_test_pos) + list(y_test_neg)

        print("Check final sizing of dataset")
        print(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
        print(f"y_train: {len(y_train)}, y_test: {len(y_test)}", end='\n')

        # --------------------------------------------------------------------------------------------------
        # 4. Make the data into dataframe format to use with tf.keras.preprocessing.image.ImageDataGenerator
        # --------------------------------------------------------------------------------------------------
        df_train = pd.DataFrame([X_train, y_train]).transpose()
        df_train.columns = ['filename', 'class']
        df_train['class'] = df_train['class'].astype(str)
        df_train = df_train.sample(frac=1)
        df_test = pd.DataFrame([X_test, y_test]).transpose()
        df_test.columns = ['filename', 'class']
        df_test['class'] = df_test['class'].astype(str)
        df_test = df_test.sample(frac=1)
        print("Final dataframe:")
        print(df_train.head(5))
        print(df_train.dtypes)

        return df_train, df_test, pd.DataFrame(), None

    def create_spectrograms(self):
        """Simple function to look at directories of positive or negative examples and convert to spectrograms
        Good if you don't need to split long audio recordings up using metadata.
        """
        for root, _, files in os.walk(self.cfg['paths']['wav_path']):
            for file in files:
                if file.lower().endswith('.flac') or file.lower().endswith('.wav'):
                    data, _ = audio.load_audio_file(Path(os.path.join(root, file)))
                    freq, time, spec = spectrogram.create_spectrogram_scipy(data, self.cfg)

                    label_path = os.path.join(self.cfg['paths']['output_path'], os.path.split(root)[-1])
                    if not os.path.exists(label_path):
                        Path(label_path).mkdir(parents=True)

                    file_name = os.path.join(label_path, os.path.splitext(file)[0] + '.jpg')
                    spectrogram.save_spectrogram(spec=spec, freq=freq, time=time, spectrogram_path=Path(file_name))
