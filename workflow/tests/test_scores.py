from __future__ import absolute_import

import math

import pytest

import numpy as np

from ..scores import precision, recall, mad_radius, mad_center
from ..scores.ospa import ospa_single
from ..scores.scp import scp_single

x = [(1, 1, 1)]
x2 = [(1, 1, 2)]
y = [(1, 3, 1)]
z = x + y


def test_mask_detection():
    shape = (10, 10)
    # Perfect match
    assert scp_single(x, x, shape) == (0, 1, 1)
    # No match
    assert scp_single(x, y, shape) == (2, 1, 1)
    assert scp_single(x, x2, shape)[0] > 0
    # 1 match, 1 miss
    assert scp_single(x, z, shape) == (1, 1, 2)
    # 1 empty, 1 not
    assert scp_single(x, [], shape) == (1, 1, 0)
    assert scp_single([], x, shape) == (1, 0, 1)
    # 2 empty arrays
    assert scp_single([], [], shape) == (0, 0, 0)


def test_ospa_single():
    # Perfect match
    assert ospa_single(x, x) == 0
    # No match
    assert ospa_single(x, y) == 1
    assert ospa_single(x, x2) > 0
    # Miss or wrong radius
    assert ospa_single(x, z) == 0.5
    # An empty array with a non empty one
    assert ospa_single(x, []) == 1
    assert ospa_single([], x) == 1
    assert ospa_single(z, []) == 1
    # Two empty arrays should match
    assert ospa_single([], []) == 0


def test_precision_recall():

    # perfect match
    y_true = [[(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(1, 1, 1), (3, 3, 1)]]
    assert precision(y_true, y_pred) == 1
    assert recall(y_true, y_pred) == 1
    assert mad_radius(y_true, y_pred) == 0
    assert mad_center(y_true, y_pred) == 0

    # partly perfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1), (5, 5, 1)]]
    assert precision(y_true, y_pred) == 0.5
    assert recall(y_true, y_pred) == 0.25
    assert mad_radius(y_true, y_pred) == 0
    assert mad_center(y_true, y_pred) == 0

    # imperfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1.2, 1.2), (3, 3, 1)]]
    assert precision(y_true, y_pred) == 1
    assert recall(y_true, y_pred) == 0.5
    assert mad_radius(y_true, y_pred) == pytest.approx(0.1)
    assert mad_center(y_true, y_pred) == pytest.approx(0.1)

    # no match
    y_true = [[(1, 1, 1)]]
    y_pred = [[(3, 3, 1)]]
    assert precision(y_true, y_pred) == 0
    assert recall(y_true, y_pred) == 0
    assert math.isnan(mad_radius(y_true, y_pred))
    assert math.isnan(mad_center(y_true, y_pred))


def test_average_precision():
    from ..scores import AveragePrecision
    ap = AveragePrecision()

    # perfect match
    y_true = [[(1, 1, 1), (3, 3, 1)]]
    y_pred = [[(1, 1, 1, 1), (1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 1

    # imperfect match
    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1, 1), (1, 5, 5, 1)]]
    assert ap(y_true, y_pred) == pytest.approx(3. / 2 / 11, rel=1e-5)
    # would be 0.125 (1 / 8) exact method

    y_true = [[(1, 1, 1), (3, 3, 1), (7, 7, 1), (9, 9, 1)]]
    y_pred = [[(1, 1, 1.2, 1.2), (1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == pytest.approx(6. / 11, rel=1e-5)
    # would be 0.5 with exact method

    # no match
    y_true = [[(1, 1, 1)]]
    y_pred = [[(1, 3, 3, 1)]]
    assert ap(y_true, y_pred) == 0
