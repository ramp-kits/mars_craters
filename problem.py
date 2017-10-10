import imp
import os

import numpy as np
import pandas as pd
# import rampwf as rw


local_workflow = imp.load_package('workflow', './workflow')


problem_title = 'Mars craters detection and classification'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = local_workflow.predictions.Predictions

# An object implementing the workflow
workflow = local_workflow.workflow.ObjectDetector()

score_types = [
    local_workflow.scores.SCP(precision=4),
    local_workflow.scores.OSPA(precision=4),
    local_workflow.scores.AveragePrecision(precision=4),
    local_workflow.scores.Precision(precision=4),
    local_workflow.scores.Recall(precision=4),
]


def get_cv(X, y):
    # _, X = folder_X
    # cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    # return cv.split(X, y)

    # quadrangles for training have shape:  4082, 3397, 2698 (total 10177)
    # for each fold use one quadrangle as test set, the other two as training
    return [(slice(0, 4082 + 3397), slice(4082 + 3397, None)),
            (slice(4082, None), slice(None, 4082)),
            (np.r_[0:4082, 4082+3397:10177], slice(4082, 4082 + 3397))]


def _read_data(path, typ):

    data_path = os.path.join(path, 'data', 'data_{}.npy'.format(typ))
    src = np.load(data_path, mmap_mode='r')

    labels_path = os.path.join(path, 'data', 'labels_{}.csv'.format(typ))
    labels = pd.read_csv(labels_path)

    # convert the dataframe with crater positions to list of
    # list of (x, y, radius) tuples (list of arrays of shape (n, 3) with n
    # true craters on an image

    # determine locations of craters for each patch in the labels array
    n_true_patches = labels.groupby('i').size().reindex(
        range(-1, src.shape[0]), fill_value=0).values
    n_cum = np.array(n_true_patches).cumsum()

    labels_array = labels[['row_p', 'col_p', 'radius_p']].values
    y = [[tuple(x) for x in labels_array[i:j]] for i, j in
         zip(n_cum[:-1], n_cum[1:])]

    # return src, y
    return src, y


def get_test_data(path='.'):
    return _read_data(path, 'test')


def get_train_data(path='.'):
    return _read_data(path, 'train')
