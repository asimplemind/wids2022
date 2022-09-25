"""Models training and evaluating"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from acoustic_ml import data_loader, data_loader_binary
from acoustic_ml import visualization
from acoustic_ml import model_fn


def train_model(cfg: dict) -> None:
    """Train model

    Arguemnts:
        cfg <dict>: configuration dictionary containing parameters needed for training the model
    Returns: None
    """

    if cfg['model']['class_mode'] == 'categorical':
        print("Create train/val/test sets")
        df_train, df_val, df_test, class_labels = data_loader.DataLoader(cfg).create_train_test_dataset()
    elif cfg['model']['class_mode'] == 'binary':
        df_train, df_val, df_test, class_labels = data_loader_binary.DataLoaderBinary(cfg).create_train_test_dataset()
    else:
        print(f"Data_loader is not implemented yet")
        sys.exit()

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)  # ,validation_split=0.1)
    train_gen = data_gen.flow_from_dataframe(dataframe=df_train,
                                             directory=None,
                                             target_size=cfg['spectrogram']['image_shape'][0:2],
                                             classes=None,
                                             class_mode=cfg['model']['class_mode'],
                                             batch_size=cfg['model']['batch_size'],
                                             shuffle=True)
    val_gen = data_gen.flow_from_dataframe(dataframe=df_val,
                                           directory=None,
                                           target_size=cfg['spectrogram']['image_shape'][0:2],
                                           classes=None,
                                           class_mode=cfg['model']['class_mode'],
                                           batch_size=cfg['model']['batch_size'],
                                           shuffle=True)

    # ImageDataGenerator infer classes from categorical model. We can use actual class labels, or use
    # attribute class_indices to obtain the class labels
    if cfg['model']['class_mode'] == 'categorical':
        # reverse class_indices
        indices_to_class = dict()
        class_labels_dict = val_gen.class_indices
        for k in class_labels_dict:
            indices_to_class[class_labels_dict[k]] = k
        del class_labels_dict
        print(f'Available class indices {indices_to_class}')

        # save classes for later use
        if not os.path.exists(os.path.join(cfg['paths']['output_path'], cfg['paths']['checkpoint_path'])):
            Path(cfg['paths']['output_path'], cfg['paths']['checkpoint_path']).mkdir(parents=True)

        print(f'Input class index {indices_to_class.values()}, saving to class_labels.npy file')
        np.save(os.path.join(cfg['paths']['output_path'], cfg['paths']['checkpoint_path'], 'class_labels.npy'),
                list(indices_to_class.values()))
        print(f'Input class index {indices_to_class.values()}, saved to class_labels.npy file\n')

    model = model_fn.build_model(cfg)
    model = model_fn.compile_model(cfg, model)

    check_generator(train_gen)
    check_generator(val_gen)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=cfg['model']['patience']),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(cfg['paths']['output_path'], cfg['paths']['checkpoint_path']),
                                           monitor='val_loss',
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           save_freq='epoch')
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg['model']['epochs'],
        callbacks=callbacks
    )

    # TODO: remove debug hooks
    # debug: test evaluate after training - comment out model.fit above to avoid wasting time
    # history = None
    # model = load_saved_model(cfg)

    if cfg['evaluate_after_training']:
        # Evaluate model via confusion matrix (fig to be saved in the same directory level the model is being run)
        # Using validation data, could add to code to create a separate final evaluation set
        print(f"Evaluate with validation data after training")

        y_pred, y_truth, labels = list(), list(), list()
        for i in range(val_gen.__len__()):
            sample = val_gen.__next__()

            if cfg['model']['class_mode'] == 'binary':
                y_truth.extend(['presence' if s > 0.5 else 'not presence' for s in sample[1]])
                predicted_prob = model.predict(sample[0])
                predicted_label = ['presence' if p > 0.5 else 'not presence' for p in predicted_prob]
                y_pred.extend(predicted_label)
                labels = ['presence', 'not presence']

            elif cfg['model']['class_mode'] == 'categorical':
                y_truth.extend([indices_to_class[np.argmax(arr)] for arr in sample[1]])
                predicted_prob = model.predict(sample[0])
                predicted_label = [indices_to_class[np.argmax(arr)] for arr in predicted_prob]
                y_pred.extend(predicted_label)
                labels = class_labels

            else:
                print('confusion matrix for none binary  is not yet implemented')
                sys.exit(0)

        visualize = visualization.Visualization
        print(f'Plotting confusion matrix')
        visualize.plot_confusion_matrix(y_truth, y_pred, labels=labels)

        print(f'Plotting train / val loss history')
        visualize.plot_loss_history(history)

    return history


def check_generator(gen) -> None:
    """check_generator Check that the generator is created correctly and print out summary

    Arguemnts:
        gen: a data generator python object
    """
    data, label = gen.__getitem__(1)
    print('label for a batch:\n', label)
    print('training data shape:', data[0].shape)
    print('training number of batches:', gen.__len__())
