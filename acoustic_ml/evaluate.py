"""Models evaluating"""
import os
import sys
import numpy as np
import cv2

import tensorflow as tf


def predict_with_model(cfg: dict) -> None:
    """Predict_with_model, this function take in test data and make inference

    Arguments:
        cfg <dict>: configuration dictionary containing parameters for making inference
    Returns: None, just printing out prediction (can be output to .csv file)
    """
    # load saved model
    saved_model = load_saved_model(cfg)

    # Predict each image and output prediction class and prediction score (consider saving to a .csv file)
    for root, _, files in os.walk(cfg['test_path']):
        for file in files:
            if file.lower().endswith('.png') or file.lower().endswith('jpg'):
                sample = cv2.imread(os.path.join(root, file)) / 255

                if cfg['model']['class_mode'] == 'binary':
                    predicted_prob = saved_model.predict(np.expand_dims(sample, axis=0), verbose=2)
                    predicted_label = 'positive' if predicted_prob[0] > 0.5 else 'negative'
                    print(f'Model predicts {file} is {predicted_label}, probability score {predicted_prob[0]}')

                elif cfg['model']['class_mode'] == 'categorical':
                    # Try loading class_labels.npy file, so we can extract the classes evaluated
                    try:
                        class_labels = np.load(os.path.join(cfg['paths']['output_path'],
                                                            cfg['paths']['checkpoint_path'], 'class_labels.npy'))
                    except Exception as e:
                        print(f'Can not load class labels, error msg: {e}',
                              f'Check .npy file exists at '
                              f'{os.path.join(cfg["paths"]["output_path"], cfg["paths"]["checkpoint_path"])}')
                        sys.exit(0)

                    # If class_labels exists, make prediction
                    predicted_prob = saved_model.predict(np.expand_dims(sample, axis=0), verbose=2)
                    predicted_index = np.argmax(predicted_prob)
                    predicted_label = class_labels[predicted_index]
                    prob = '{:.4f}'.format(predicted_prob[0][predicted_index])
                    print(f'Model predicts {file} is {predicted_label}, probability score {prob}')

                else:
                    print('predict with model for this class mode outside of '
                          'binary or categorical are not yet implemented')


def load_saved_model(cfg: dict):
    """load_saved_model, this function load the saved model based on the path defined in the configuration file

    Arguments:
        cfg <dict>: configuration dictionary containing all configurable parameters
    Returns:
        saved_model <keras.Sequential>: keras model
    """
    print('Loading model from ', os.path.join(cfg['paths']['output_path'], cfg['paths']['checkpoint_path']))
    saved_model = tf.keras.models.load_model(os.path.join(cfg['paths']['output_path'], cfg['paths']['checkpoint_path']))
    print('Finished loading saved model')
    return saved_model
