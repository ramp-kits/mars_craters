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
    local_workflow.scores.Ospa(),
    local_workflow.scores.Precision(),
    local_workflow.scores.Recall(),
    local_workflow.scores.MAD_Center(),
    local_workflow.scores.MAD_Radius(),
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
    # list of (x, y, radius) tuples
    # -> for each patch number, select the corresponding craters and convert
    #    the x_p, y_p and radius_p columns to tuples
    y = [list(labels
              .loc[labels.id == '77_{0}'.format(i), ['x_p', 'y_p', 'radius_p']]
              .itertuples(name=None, index=False))
         for i in range(src.shape[0])]

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
