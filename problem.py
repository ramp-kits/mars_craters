import imp
import os

import numpy as np
import pandas as pd

import rampwf as rw


problem_title = 'Mars craters detection and classification'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_detection()
# An object implementing the workflow
workflow = rw.workflows.ObjectDetector()

score_types = [
    rw.score_types.SCP(
        shape=(224, 224), precision=4, minipatch=[56, 168, 56, 168]),
    # rw.score_types.OSPA(precision=4, minipatch=[56, 168, 56, 168]),
    rw.score_types.OSPA(precision=4),
    rw.score_types.AverageDetectionPrecision(name='ap', precision=4),
    rw.score_types.DetectionPrecision(name='prec', precision=4),
    rw.score_types.DetectionRecall(name='rec', precision=4),
]


def get_cv(X, y):
    # 3 quadrangles for training have not exactly the same size,
    # but for simplicity just cut in 3
    # for each fold use one quadrangle as test set, the other two as training

    n_tot = len(X)
    n1 = n_tot // 3
    n2 = n1 * 2

    return [(slice(0, n2), slice(n2, None)),
            (slice(n1, None), slice(None, n1)),
            (np.r_[0:n1, n2:n_tot], slice(n1, n2))]


def _read_data(path, typ):
    """
    Read and process data and labels

    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'train', 'test'}

    Returns
    -------
    X, y data

    """
    try:
        data_path = os.path.join(path, 'data', 'data_{}.npy'.format(typ))
        src = np.load(data_path, mmap_mode='r')

        labels_path = os.path.join(path, 'data', 'labels_{}.csv'.format(typ))
        labels = pd.read_csv(labels_path)
    except IOError:
        raise IOError("'data/data_{}.npy' and 'data/labels_{}.csv' are not "
                      "found. Ensure you ran 'python download_data.py' to "
                      "obtain the train/test data".format(typ))

    # convert the dataframe with crater positions to list of
    # list of (x, y, radius) tuples (list of arrays of shape (n, 3) with n
    # true craters on an image

    # determine locations of craters for each patch in the labels array
    n_true_patches = labels.groupby('i').size().reindex(
        range(src.shape[0]), fill_value=0).values
    # make cumulative sum to obtain start/stop to slice the labels
    n_cum = np.array(n_true_patches).cumsum()
    n_cum = np.insert(n_cum, 0, 0)

    labels_array = labels[['row_p', 'col_p', 'radius_p']].values
    y = [[tuple(x) for x in labels_array[i:j]]
         for i, j in zip(n_cum[:-1], n_cum[1:])]
    # convert list to object array of lists
    y_array = np.empty(len(y), dtype=object)
    y_array[:] = y

    # return src, y
    return src, y_array


def get_test_data(path='.'):
    return _read_data(path, 'test')


def get_train_data(path='.'):
    return _read_data(path, 'train')
