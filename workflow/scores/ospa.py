from __future__ import division

import numpy as np

from .detection_base import DetectionBaseScoreType
from .precision_recall import _match_tuples
from .precision_recall import _select_minipatch_tuples


def ospa_single(y_true, y_pred, cut_off=1, minipatch=None):
    """
    OSPA score on single patch. See docstring of `ospa` for more info.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (3, x)
        arrays of (x, y, radius)
    cut_off : float, optional (default is 1)
        penalizing value for wrong cardinality
    minipatch : list of int, optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float: distance between input arrays

    """
    n_true = len(y_true)
    n_pred = len(y_pred)

    n_max = max(n_true, n_pred)
    n_min = min(n_true, n_pred)

    # No craters and none found
    if n_true == 0 and n_pred == 0:
        return 0

    # No craters and some found or existing craters but non found
    if n_true == 0 or n_pred == 0:
        return cut_off

    # OSPA METRIC
    id_true, id_pred, ious = _match_tuples(y_true, y_pred)

    if minipatch is not None:
        true_in_minipatch = _select_minipatch_tuples(y_true[id_true])
        pred_in_minipatch = _select_minipatch_tuples(y_pred[id_pred])
        is_valid = true_in_minipatch & pred_in_minipatch
        iou_score = ious[is_valid].sum()
    else:
        iou_score = ious.sum()

    distance_score = n_min - iou_score
    cardinality_score = cut_off * (n_max - n_min)

    dist = 1 / n_max * (distance_score + cardinality_score)

    return dist


class OSPA(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ospa', precision=2, conf_threshold=0.5,
                 cut_off=1, minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch
        self.cut_off = cut_off

    def detection_score(self, y_true, y_pred):
        """Optimal Subpattern Assignment (OSPA) metric for IoU score.

        This metric provides a coherent way to compute the miss-distance
        between the detection and alignment of objects. Among all
        combinations of true/predicted pairs, if finds the best alignment
        to minimise the distance, and still takes into account missing
        or in-excess predicted values through a cardinality score.

        The lower the value the smaller the distance.

        Parameters
        ----------
        y_true, y_pred : list of list of tuples

        Returns
        -------
        float: distance between input arrays

        References
        ----------
        http://www.dominic.schuhmacher.name/papers/ospa.pdf

        """
        scores = [ospa_single(t, p, self.cut_off, self.minipatch)
                  for t, p in zip(y_true, y_pred)]
        weights = [len(t) for t in y_true]
        return np.average(scores, weights=weights)
