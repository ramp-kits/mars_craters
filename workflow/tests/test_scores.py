from __future__ import absolute_import

import math

import pytest

import numpy as np

from ..scores import (ospa, score_craters_on_patch, precision, recall,
                      mad_radius, mad_center)

x = [(1, 1, 1)]
x2 = [(1, 1, 2)]
y = [(1, 3, 1)]
z = x + y


def test_score_craters_on_patch():
    # Perfect match
    assert score_craters_on_patch(x, x) == 1
    # No match
    assert score_craters_on_patch(x, y) == 0
    assert score_craters_on_patch(x, x2) < 1
    # 1 match, 1 miss
    assert score_craters_on_patch(x, z) == 0.5
    # 1 empty, 1 not
    assert score_craters_on_patch(x, []) == 0
    assert score_craters_on_patch([], x) == 0
    # 2 empty arrays
    assert score_craters_on_patch([], []) == 1


def test_ospa():
    x_arr = np.array(x).T
    x2_arr = np.array(x2).T
    y_arr = np.array(y).T
    z_arr = np.array(z).T
    empty_arr = np.atleast_2d([]).T

    # Match
    assert ospa(x_arr, x_arr) == 0
    # Miss or wrong radius
    assert ospa(x_arr, y_arr) == 1
    assert ospa(x_arr, x2_arr) > 0
    # One match, one miss
    assert ospa(x_arr, z_arr) - 0.5 < 1e-6
    # An empty array with a non empty one
    assert ospa(x_arr, empty_arr) == 1
    assert ospa(empty_arr, x_arr) == 1
    assert ospa(z_arr, empty_arr) == 1
    # Two empty arrays should match
    assert ospa(empty_arr, empty_arr) == 0


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
