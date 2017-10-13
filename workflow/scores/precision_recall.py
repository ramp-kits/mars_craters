from __future__ import division

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection_base import DetectionBaseScoreType

from ..iou import cc_iou as iou


def _select_minipatch_tuples(y_list, minipatch):
    """

    Parameters
    ----------
    y_list : list of tuples
        Full list of labels and predictions
    minipatch : list of int
        Bounds of the internal scoring patch

    Returns
    -------
    y_list_cut : list of bool
        List of booleans corresponding to whether the circle is in
        the minipatch or not

    """
    row_min, row_max, col_min, col_max = minipatch

    y_list = np.asarray(y_list)

    y_list_cut = ((y_list[0] >= col_min) & (y_list[0] < col_max) &
                  (y_list[1] >= row_min) & (y_list[1] < row_max))

    return y_list_cut


def _match_tuples(y_true, y_pred):
    """
    Given set of true and predicted (x, y, r) tuples, determine the best
    possible match.

    Parameters
    ----------
    y_true, y_pred : list of tuples

    Returns
    -------
    (idxs_true, idxs_pred, ious)
        idxs_true, idxs_pred : indices into y_true and y_pred of matches
        ious : corresponding IOU value of each match

        The length of the 3 arrays is identical and the minimum of the length
        of y_true and y_pred

    """

    n_true = len(y_true)
    n_pred = len(y_pred)

    iou_matrix = np.empty((n_true, n_pred))

    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = iou(y_true[i], y_pred[j])

    idxs_true, idxs_pred = linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]
    return idxs_true, idxs_pred, ious


def _count_matches(y_true, y_pred, matches, iou_threshold=0.5):
    """
    Count the number of matches.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float

    Returns
    -------
    (n_true, n_pred_all, n_pred_correct):
        Number of true craters
        Number of total predicted craters
        Number of correctly predicted craters

    """
    val_numbers = []

    for y_true_p, y_pred_p, match_p in zip(y_true, y_pred, matches):
        n_true = len(y_true_p)
        n_pred = len(y_pred_p)

        _, _, ious = match_p
        p = (ious >= iou_threshold).sum()

        val_numbers.append((n_true, n_pred, p))

    n_true, n_pred_all, n_pred_correct = np.array(val_numbers).sum(axis=0)

    return n_true, n_pred_all, n_pred_correct


def _locate_matches(y_true, y_pred, matches, iou_threshold=0.5):
    """
    Given list of list of matching craters, return contiguous array of all
    craters x, y, r.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float

    Returns
    -------
    loc_true, loc_pred
        Each is 2D array (n_patches, 3) with x, y, r columns

    """
    loc_true = []
    loc_pred = []

    for y_true_p, y_pred_p, matches_p in zip(y_true, y_pred, matches):

        for idx_true, idx_pred, iou_val in zip(*matches_p):
            if iou_val >= iou_threshold:
                loc_true.append(y_true_p[idx_true])
                loc_pred.append(y_pred_p[idx_pred])

    if loc_true:
        return np.array(loc_true), np.array(loc_pred)
    else:
        return np.empty((0, 3)), np.empty((0, 3))


def precision(y_true, y_pred, matches=None, iou_threshold=0.5,
              minipatch=None):
    """
    Precision score (fraction of correct predictions).

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match

    Returns
    -------
    precision_score : float [0 - 1]
    """
    if matches is None:
        matches = [_match_tuples(t, p)
                   for t, p in zip(y_true, y_pred)]

    n_true, n_pred_all, n_pred_correct = _count_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    return n_pred_correct / n_pred_all


def recall(y_true, y_pred, matches=None, iou_threshold=0.5,
           minipatch=None):
    """
    Recall score (fraction of true objects that are predicted).

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match

    Returns
    -------
    recall_score : float [0 - 1]
    """
    if matches is None:
        matches = [_match_tuples(t, p)
                   for t, p in zip(y_true, y_pred)]

    n_true, n_pred_all, n_pred_correct = _count_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    return n_pred_correct / n_true


def mad_radius(y_true, y_pred, matches=None, iou_threshold=0.5,
               minipatch=None):
    """
    Relative Mean absolute deviation (MAD) of the radius.

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match

    Returns
    -------
    mad_radius : float > 0
    """
    if matches is None:
        matches = [_match_tuples(t, p)
                   for t, p in zip(y_true, y_pred)]

    loc_true, loc_pred = _locate_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    return np.abs((loc_pred[:, 2] - loc_true[:, 2]) / loc_true[:, 2]).mean()


def mad_center(y_true, y_pred, matches=None, iou_threshold=0.5,
               minipatch=None):
    """
    Relative Mean absolute deviation (MAD) of the center (relative to the
    radius of the true object).

    Parameters
    ----------
    y_true, y_pred : list of list of tuples
    matches : optional, output of _match_tuples
    iou_threshold : float
        Threshold to determine match

    Returns
    -------
    mad_center : float > 0
    """
    if matches is None:
        matches = [_match_tuples(t, p)
                   for t, p in zip(y_true, y_pred)]

    loc_true, loc_pred = _locate_matches(
        y_true, y_pred, matches, iou_threshold=iou_threshold)

    d = np.sqrt((loc_pred[:, 0] - loc_true[:, 0]) ** 2 + (
        loc_pred[:, 1] - loc_true[:, 1]) ** 2)

    return np.abs(d / loc_true[:, 2]).mean()


# ScoreType classes


class Precision(DetectionBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='precision', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return precision(y_true, y_pred, minipatch=self.minipatch)


class Recall(DetectionBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='recall', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return recall(y_true, y_pred, minipatch=self.minipatch)


class MAD_Center(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mad_center', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return mad_center(y_true, y_pred, minipatch=self.minipatch)


class MAD_Radius(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mad_radius', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        return mad_radius(y_true, y_pred, minipatch=self.minipatch)
