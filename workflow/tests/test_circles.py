import pytest
import numpy as np

from ..scores._circles import project_circle, circle_map

circle = (1, 1, 1)
x = [circle]


def test_project_circle():
    shape = (10, 10)
    assert project_circle(circle, image=np.zeros(shape)).shape == shape
    assert project_circle(circle, shape=shape).shape == shape

    classic = project_circle(circle, shape=shape)
    assert classic.min() == 0
    assert classic.max() == 1

    negative = project_circle(circle, shape=shape, negative=True)
    assert negative.min() == -1
    assert negative.max() == 0

    normalized = project_circle(circle, shape=shape, normalize=True)
    assert normalized.min() == 0
    assert normalized.sum() == 1

    normalized_neg = project_circle(circle, shape=shape,
                                    normalize=True, negative=True)
    assert normalized_neg.max() == 0
    assert normalized_neg.sum() == -1

    with pytest.raises(ValueError):
        project_circle(circle)
        project_circle(circle, image=None, shape=None)


def test_circle_map():
    assert circle_map([], []).max() == 0
    assert circle_map([], []).sum() == 0
    assert circle_map(x, []).sum() == 1
    assert circle_map([], x).sum() == -1
    assert circle_map(x, x).sum() == 0
