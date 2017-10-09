import imp

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


def get_cv(folder_X, y):
    # _, X = folder_X
    # cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    # return cv.split(X, y)

    # for now dummy CV that has one fold with all data for both train/valid
    return [(slice(None), slice(None)), ]


def _read_data(path, f_name):
    src = np.load('data/images_quad_77.npy', mmap_mode='r')
    labels = pd.read_csv("data/quad77_labels.csv")

    # convert the dataframe with crater positions to list of
    # list of (x, y, radius) tuples (list of arrays of shape (n, 3) with n
    # true craters on an image

    # TODO this will not be needed if we appropriately save labels csv file
    labels['i'] = labels.id.str[3:].astype(int)
    labels = labels.sort_values('i')

    # determine locations of craters for each patch in the labels array
    n_true_patches = labels.groupby('i').size().reindex(
        range(-1, src.shape[0]), fill_value=0).values
    n_cum = np.array(n_true_patches).cumsum()

    labels_array = labels[['x_p', 'y_p', 'radius_p']].values
    y = [[tuple(x) for x in labels_array[i:j]] for i, j in
         zip(n_cum[:-1], n_cum[1:])]

    # df = pd.read_csv(os.path.join(path, 'data', f_name))
    # X = df['id'].values
    # y = df['class'].values
    # folder = os.path.join(path, 'data', 'imgs')

    # return src, y
    return src[:200, :, :], y[:200]


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
