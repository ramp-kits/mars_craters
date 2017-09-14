from __future__ import division

import itertools
import numpy as np

from .iou import cc_iou as iou


def score_craters_on_patch(y_true, y_pred):
    """
    Main OSPA score for single patch

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch

    Returns
    -------
    float : score for a given path, the higher the better

    """
    y_true = np.atleast_2d(y_true).T
    y_pred = np.atleast_2d(y_pred).T

    ospa_score = ospa(y_true, y_pred)

    score = 1 - ospa_score

    return score


def score_iou(y_true, y_pred):
    pass


def score_completeness(y_true, y_pred):
    pass


def ospa(x_arr, y_arr, cut_off=1):
    """
    Optimal Subpattern Assignment (OSPA) metric for IoU score

    This metric provides a coherent way to compute the miss-distance
    between the detection and alignment of objects. Among all
    combinations of true/predicted pairs, if finds the best alignment
    to minimise the distance, and still takes into account missing
    or in-excess predicted values through a cardinality score.

    The lower the value the smaller the distance.

    Parameters
    ----------
    x_arr, y_arr : ndarray of shape (3, x)
        arrays of (x, y, radius)
    cut_off : float, optional (default is 1)
        penalizing value for wrong cardinality

    Returns
    -------
    float: distance between input arrays

    References
    ----------
    http://www.dominic.schuhmacher.name/papers/ospa.pdf

    """
    x_size = x_arr.size
    y_size = y_arr.size

    _, m = x_arr.shape
    _, n = y_arr.shape

    if m > n:
        return ospa(y_arr, x_arr, cut_off)

    # NO CRATERS
    # ----------
    # GOOD MATCH
    if x_size == 0 and y_size == 0:
        return 0

    # BAD MATCH
    if x_size == 0 or y_size == 0:
        return cut_off

    # CRATERS
    # -------
    # TOO MANY OR TOO FEW DETECTIONS
    # ARBITRARY THRESHOLD TO SAVE COMPUTING TIME
    if n > 4 * m and n > 15:
        return cut_off

    # OSPA METRIC
    iou_score = 0
    permutation_indices = itertools.permutations(range(n), m)
    for idx in permutation_indices:
        new_dist = sum(iou(x_arr[:, j], y_arr[:, idx[j]])
                       for j in range(m))
        iou_score = max(iou_score, new_dist)

    distance_score = m - iou_score
    cardinality_score = cut_off * (n - m)

    dist = 1 / n * (distance_score + cardinality_score)

    return dist
