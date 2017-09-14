import os
import pandas as pd
# import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

import local_workflow
import local_scores

problem_title = 'Mars craters detection and classification'

# _target_column_name = 'class'
# _prediction_label_names = [0, 1, 2, 3]
# # A type (class) which will be used to create wrapper objects for y_pred
# Predictions = rw.prediction_types.make_multiclass(
#     label_names=_prediction_label_names)
# An object implementing the workflow
workflow = local_workflow.ObjectDetector(
    test_batch_size=16,
    chunk_size=256,
    n_jobs=8,
    # img_file_extension='png',
    # n_classes=len(_prediction_label_names),
)

score_types = [
    local_scores.Accuracy(name='acc'),
    local_scores.NegativeLogLikelihood(name='nll'),
]


def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values
    folder = os.path.join(path, 'data', 'imgs')
    return (folder, X), y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
