from __future__ import division

import numpy as np

from ._draw import circle as circle_coords


def project_circle(circle, image=None, shape=None,
                   normalize=True, negative=False):
    """
    Project circles on an image.

    Parameters
    ----------
    circle : array-like
        x, y, radius
    image : array-like, optional
        image on which to project the circle
    shape : tuple of ints, optional
        shape of the image on which to project the circle
    normalize : bool, optional (default is `True`)
        normalize the total surface of the circle to unity
    negative : bool, optional (default is `False`)
        subtract the circle instead of adding it

    Returns
    -------
    array-like : image with projected circle

    """
    if image is None:
        if shape is None:
            raise ValueError("Either `image` or `shape` must be defined")
        else:
            image = np.zeros(shape)
    else:
        shape = image.shape
    x, y, radius = circle
    coords = circle_coords(x, y, radius, shape=shape)
    value = 1
    if normalize:
        value /= coords[0].size
    if negative:
        value = -value
    image[coords] += value
    return image


def circle_maps(y_true, y_pred, shape):
    """
    Create a map to compare true and predicted craters.

    The craters (circles) are projected on the map with a coefficient
    chosen so its sum is normalized to unity.

    True and predicted craters are projected with a different sign,
    so that good predictions tend to cancel out the true craters.

    Parameters
    ----------
    y_pred, y_true : array-like of shape (3, X)
        list of circle positions (x, y, radius)
    shape : tuple of ints, optional
        shape of image

    Returns
    -------
    array-like : image with projected true and predicted circles

    """
    map_true = np.zeros(shape)
    map_pred = np.zeros(shape)

    # Add true craters positively
    for circle in y_true:
        map_true = project_circle(
            circle, map_true, shape=shape, normalize=True)

    # Add predicted craters negatively
    for circle in y_pred:
        map_pred = project_circle(
            circle, map_pred, shape=shape, normalize=True)

    return map_true, map_pred
