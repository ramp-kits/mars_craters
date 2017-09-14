import math
import pytest

from .. import iou


def test_cc_iou():
    circle1 = (0, 0, 1)
    circle2 = (0, 4, 1)
    circle3 = (1, 1, 2)
    circle1_2 = (0, 0, 2)
    assert iou.cc_iou(circle1, circle1) - 1 < 1e-6
    assert iou.cc_iou(circle1, circle2) < 1e-6
    assert iou.cc_iou(circle2, circle1) < 1e-6
    assert iou.cc_iou(circle1, circle3) - math.pi < 1e-6
    assert iou.cc_iou(circle3, circle1) - math.pi < 1e-6
    assert iou.cc_iou(circle1_2, circle1) == 0.25
    assert iou.cc_iou(circle1, circle1_2) == 0.25


def test_cc_intersection():
    # Zero distance
    assert iou.cc_intersection(0, 1, 2) - 4 * math.pi < 1e-6

    # Zero radius
    assert iou.cc_intersection(1, 0, 1) == 0
    assert iou.cc_intersection(1, 1, 0) == 0

    # High distance
    assert iou.cc_intersection(4, 1, 2) == 0

    # Classic test
    assert iou.cc_intersection(1, 1, 2) - math.pi < 1e-6

    with pytest.raises(ValueError):
        iou.cc_intersection(-1, 1, 1)
        iou.cc_intersection(1, -1, 1)
        iou.cc_intersection(1, 1, -1)
