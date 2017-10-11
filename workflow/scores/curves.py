from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from rampwf.score_types.base import BaseScoreType


def _filter_y_pred(y_pred, conf_threshold):
    return [[detected_object[1:] for detected_object in y_pred_patch
             if detected_object[0] > conf_threshold]
            for y_pred_patch in y_pred]


def precision_recall_curve(y_true, y_pred, conf_thresholds, iou_threshold=0.5):
    """
    Calculate precision and recall for different confidence thresholds.

    Parameters
    ----------
    y_true : list of list of tuples
        Tuples are of form (x, y, radius).
    y_pred : list of list of tuples
        Tuples are of form (x, y, radius, confidence).
    conf_thresholds : array-like
        The confidence threshold for which to calculate the
        precision and recall.
    iou_threshold : float
        Threshold to determine match.

    Returns
    -------
    ps, rs : arrays with precisions, recalls for each confidence threshold

    """
    from .precision_recall import precision, recall

    ps = []
    rs = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        ps.append(precision(
            y_true, y_pred_above_confidence, iou_threshold=iou_threshold))
        rs.append(recall(
            y_true, y_pred_above_confidence, iou_threshold=iou_threshold))

    return np.array(ps), np.array(rs)


def mask_detection_curve(y_true, y_pred, conf_thresholds):
    """
    Calculate mask detection score for different confidence thresholds.

    Parameters
    ----------
    y_true : list of list of tuples
        Tuples are of form (x, y, radius).
    y_pred : list of list of tuples
        Tuples are of form (x, y, radius, confidence).
    conf_thresholds : array-like
        The confidence threshold for which to calculate the
        precision and recall.

    Returns
    -------
    ms : array with score for each confidence threshold

    """
    from .mask import mask_detection

    ms = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        ms.append(mask_detection(y_true, y_pred_above_confidence))

    return np.array(ms)


def ospa_curve(y_true, y_pred, conf_thresholds):
    """
    Calculate OSPA score for different confidence thresholds.

    Parameters
    ----------
    y_true : list of list of tuples
        Tuples are of form (x, y, radius).
    y_pred : list of list of tuples
        Tuples are of form (x, y, radius, confidence).
    conf_thresholds : array-like
        The confidence threshold for which to calculate the
        precision and recall.

    Returns
    -------
    os : array with OSPA score for each confidence threshold

    """
    from .ospa import ospa

    os = []

    for conf_threshold in conf_thresholds:
        y_pred_above_confidence = _filter_y_pred(y_pred, conf_threshold)
        os.append(ospa(y_true, y_pred_above_confidence))

    return np.array(os)


def average_precision_interpolated(ps, rs):
    """
    The Average Precision (AP) score.

    Calculation based on the 11-point interpolation of the precision-recall
    curve (method described for Pascal VOC challenge,
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf).

    TODO: they changed this in later:
    http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

    https://stackoverflow.com/questions/36274638/map-metric-in-object-detection-and-computer-vision

    Parameters
    ----------
    ps, rs : arrays of same length with corresponding precision / recall scores

    Returns
    -------
    ap : int [0 - 1]
        Average precision score

    """
    ps = np.asarray(ps)
    rs = np.asarray(rs)

    p_at_r = []

    for r in np.arange(0, 1.1, 0.1):
        p = np.array(ps)[np.array(rs) >= r]
        if p.size:
            p_at_r.append(np.nanmax(p))
        else:
            p_at_r.append(0)

    ap = np.mean(p_at_r)
    return ap


# ScoreType classes


class AveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='average_precision', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        conf_thresholds = np.linspace(0.0, 1, 50)
        ps, rs = precision_recall_curve(y_true, y_pred, conf_thresholds)
        return average_precision_interpolated(ps, rs)


# plotting utility functions


def plot_precision_recall_curve(ps, rs):

    ap = average_precision_interpolated(ps, rs)

    fig, ax = plt.subplots()
    ax.plot(rs, ps, 'o-')
    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.text(0.7, 0.9, 'AP = {:.2f}'.format(ap), fontsize=16)

    return fig, ax


def plot_curves(ps, rs, ms, os, conf_thresholds):
    fig, ax = plt.subplots()
    ax.plot(conf_thresholds, ms, label='scp')
    ax.plot(conf_thresholds, 1 - os, label='ospa', color='C0', linestyle='--')

    ax.plot(conf_thresholds, ps, 'C1', label='precision')
    ax.plot(conf_thresholds, rs, 'C2', label='recall')
    ax.plot(conf_thresholds, 1 - (2 * (ps * rs) / (ps + rs)), 'C3',
            label='f1 score')

    ax.legend(loc=7)

    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")

    ax.set_ylim(0, 1)

    # ax.axhline(1, linestyle='--', color='grey')
    # ax.axvline(conf_thresholds[17], color='grey', linestyle='--')
    ax.spines['top'].set(linestyle='--', color='grey')
    ax.spines['right'].set(linestyle='--', color='grey')

    ax.text(0.8, 0.87,
            'AP = {:.2f}'.format(average_precision_interpolated(ps, rs)))
    ax.text(0.7, 0.80, 'min(MD) = {:.2f}'.format(np.min(ms)))
