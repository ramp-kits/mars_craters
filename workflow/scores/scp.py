from __future__ import division

import numpy as np

from .detection_base import DetectionBaseScoreType
from ._circles import circle_maps


def scp_single(y_true, y_pred, shape, minipatch=None):
    """
    L1 distance between superposing bounding box cylinder or prism maps.

    True craters are projected positively, predicted craters negatively,
    so they can cancel out. Then the sum of the absolute value of the
    residual map is taken.

    The best score value for a perfect match is 0.
    The worst score value for a given patch is the sum of all crater
    instances in both `y_true` and `y_pred`.

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch
    shape : tuple of int
        Shape of the main patch
    minipatch : list of int, optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float : score for a given patch, the lower the better

    """
    map_true, map_pred = circle_maps(y_true, y_pred, shape)
    if minipatch is not None:
        map_true = map_true[
            minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
        map_pred = map_pred[
            minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
    # Sum all the pixels
    score = np.abs(map_true - map_pred).sum()
    n_true = map_true.sum()
    n_pred = map_pred.sum()
    return score, n_true, n_pred


class SCP(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, shape, name='scp', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.shape = shape
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        """
        Score based on a matching by reprojection of craters on mask-map.

        True craters are projected positively, predicted craters negatively,
        so they can cancel out. Then the sum of the absolute value of the
        residual map is taken.

        The best score value for a perfect match is 0.
        The worst score value for a given patch is the sum of all crater
        instances in both `y_true` and `y_pred`.

        Parameters
        ----------
        y_true : list of list of tuples (x, y, radius)
            List of coordinates and radius of actual craters for set of patches
        y_pred : list of list of tuples (x, y, radius)
            List of coordinates and radius of predicted craters for set of
            patches

        Returns
        -------
        float : score for a given patch, the lower the better

        """
        scps = np.array(
            [scp_single(t, p, self.shape, self.minipatch)
             for t, p in zip(y_true, y_pred)])
        return np.sum(scps[:, 0]) / np.sum(scps[:, 1:3])
