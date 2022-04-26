import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
sns.set()


class Visualization:
    """Visualization

    This module contains different visuals for understanding the performance of the model being trained
    """
    def __init__(self):
        # nothing to initialize yet, perhaps use this as a gathering spot for filenames, color maps, styles, etc
        # or, don't even bother with a class!
        pass

    @staticmethod
    def plot_confusion_matrix(y_true: list, y_pred: list, labels: list = None,
                              title: str = "Confusion matrix", cmap: str = 'YlGnBu_r') -> None:
        """
        Arguments:
            y_true <list>: truth labels
            y_pred <list>: predicted labels
            labels <list>: class labels
            title <str>: title for the plot
            cmap <str>: color map for the confusion matrx (heatmap)
        Returns: None
        """
        if labels is None:
            labels = list(set(y_true))
            print(f'labels = {labels}')

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        sns.heatmap(cm, annot=True, cmap=cmap)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        tick_marks = np.arange(len(set(y_true)))
        ax.set_xticks(tick_marks+0.25, rotation=75, va='top', ha='left', labels=labels)
        ax.set_yticks(tick_marks+0.5, rotation=0, va='top', labels=labels)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig('confusion_matrix.png', transparent=True)
        print(f'View confusion matrix at {os.path.join(os.curdir, "confusion_matrix.png")}')
    
    @staticmethod
    def plot_loss_history(history):
        """Plot the loss history data

        Arguments:
            history <a history object> : contains information from model.fit
        Returns: None
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        metrics = history.history
        plt.plot(history.epoch, metrics['loss'])
        plt.plot(history.epoch, metrics['val_loss'], '.-')
        plt.title("Training and Validation Loss")
        plt.legend(['train loss', 'val loss'])
        fig.tight_layout()
        fig.savefig('train_val_loss.png', transparent=True)
        print(f'View train/validation loss plot at {os.path.join(os.curdir, "train_val_loss.png")}')
