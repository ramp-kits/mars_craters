from __future__ import division

import math
from itertools import repeat

import numpy as np

from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier

from skimage.feature import blob_doh
from mahotas.features import zernike_moments
from mahotas.features import surf

###############################################################################
# Extractor


class BlobExtractor(BaseEstimator):
    """Feature extractor using a blob detector.

    This extractor will detect candidate regions using a blob detector,
    i.e. maximum of the determinant of Hessian, and will extract the Zernike's
    moments and SURF descriptors for each regions.

    Parameters
    ----------
    min_radius : int, default=5
        The minimum radius of the candidate to be detected.

    max_radius : int, default=30
        The maximum radius of the candidate to be detected.

    blob_threshold : float, default=0.01
        The threshold used to extract the candidate region in the DoH map.
        Values above this threshold will be considered as a ROI.

    overlap : float, default=0.2
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than threshold, the smaller blob is eliminated.

    padding : float, default=2.0
        The region around the blob will be enlarged by the factor given in
        padding.

    iou_threshold : float, default=0.4
        A value between 0 and 1. If the IOU between the candidate and the
        target is greater than this threshold, the candidate is considered as a
        crater.

    """

    def __init__(self, min_radius=5, max_radius=30, blob_threshold=0.01,
                 overlap=0.2, padding=2.0, iou_threshold=0.4):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.blob_threshold = blob_threshold
        self.overlap = overlap
        self.padding = padding
        self.iou_threshold = iou_threshold

    def fit(self, X, y=None, **fit_params):
        # This extractor does not require any fitting
        return self

    def _extract_features(self, X, candidate):

        y, x, radius = int(candidate[0]), int(candidate[1]), candidate[2]
        padded_radius = int(self.padding * radius)

        # compute the coordinate of the patch to select
        x_min = max(x - padded_radius, 0)
        y_min = max(y - padded_radius, 0)
        x_max = min(x + padded_radius, X.shape[0] - 1)
        y_max = min(y + padded_radius, X.shape[1] - 1)

        patch = X[y_min:y_max, x_min:x_max]

        # compute Zernike moments
        zernike = zernike_moments(patch, radius)

        # compute SURF descriptor
        scale_surf = radius / self.min_radius
        keypoint = np.array([[y, x, scale_surf, 0.1, 1]])
        surf_descriptor = surf.descriptors(X, keypoint,
                                           is_integral=False).ravel()
        if not surf_descriptor.size:
            surf_descriptor = np.zeros((70, ))

        return np.hstack((zernike, surf_descriptor))

    def extract(self, X, y=None, **fit_params):
        candidate_blobs = blob_doh(X, min_sigma=self.min_radius,
                                   max_sigma=self.max_radius,
                                   threshold=self.blob_threshold,
                                   overlap=self.overlap)

        # convert the candidate to list of tuple
        candidate_blobs = [tuple(blob) for blob in candidate_blobs]

        # extract feature to be returned
        features = [self._extract_features(X, blob)
                    for blob in candidate_blobs]

        if y is None:
            # branch used during testing
            return features, candidate_blobs, [None] * len(features)
        elif not y:
            # branch if there is no crater in the image
            labels = [0] * len(candidate_blobs)

            return features, candidate_blobs, labels
        else:
            # case the we did not detect any blobs
            if not len(features):
                return ([], [], [])

            # find the maximum scores between each candidate and the
            # ground-truth
            scores_candidates = [max(map(cc_iou, repeat(blob, len(y)), y))
                                 for blob in candidate_blobs]

            # threshold the scores
            labels = [0 if score < self.iou_threshold else 1
                      for score in scores_candidates]

            return features, candidate_blobs, labels

    def fit_extract(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).extract(X)
        else:
            return self.fit(X, y, **fit_params).extract(X, y)


###############################################################################
# Detector


class ObjectDetector(object):
    """Object detector.

    Object detector using an extractor (which is used to extract feature) and
    an estimator.

    Parameters
    ----------
    extractor : object, default=BlobDetector()
        The feature extractor used before to train the estimator.

    estimator : object, default=GradientBoostingClassifier()
        The estimator used to decide if a candidate is a crater or not.

    Attributes
    ----------
    extractor_ : object,
        The actual extractor used after fit.

    estimator_ : object,
        The actual estimator used after fit.

    """

    def __init__(self, extractor=None, estimator=None):
        self.extractor = extractor
        self.estimator = estimator

    def _extract_features(self, X, y):
        # extract feature for all the image containing craters
        data_extracted = [self.extractor_.fit_extract(image, craters)
                          for image, craters in zip(X, y)]

        # organize the data to fit it inside the classifier
        data, location, target, idx_cand_to_img = [], [], [], []
        for img_idx, candidate in enumerate(data_extracted):
            # check if this is an empty features
            if len(candidate[0]):
                data.append(np.vstack(candidate[0]))
                location += candidate[1]
                target += candidate[2]
                idx_cand_to_img += [img_idx] * len(candidate[1])
        # convert to numpy array the data needed to feed the classifier
        data = np.concatenate(data)
        target = np.array(target)

        return data, location, target, idx_cand_to_img

    def fit(self, X, y):
        if self.extractor is None:
            self.extractor_ = BlobExtractor()
        else:
            self.extractor_ = clone(self.extractor)

        if self.estimator is None:
            self.estimator_ = GradientBoostingClassifier(n_estimators=100)
        else:
            self.estimator_ = clone(self.estimator)

        # extract the features for the training data
        data, _, target, _ = self._extract_features(X, y)

        # fit the underlying classifier
        self.estimator_.fit(data, target)

        return self

    def predict(self, X):
        # extract the data for the current image
        data, location, _, idx_cand_to_img = self._extract_features(
            X, [None] * len(X))

        # classify each candidate
        y_pred = self.estimator_.predict_proba(data)

        # organize the output
        output = [[] for _ in range(len(X))]
        crater_idx = np.flatnonzero(self.estimator_.classes_ == 1)[0]
        for crater, pred, img_idx in zip(location, y_pred, idx_cand_to_img):
            output[img_idx].append((pred[crater_idx],
                                    crater[0], crater[1], crater[2]))

        return np.array(output, dtype=object)

###############################################################################
# IOU function


def cc_iou(circle1, circle2):
    """
    Intersection over Union (IoU) between two circles

    Parameters
    ----------
    circle1 : tuple of floats
        first circle parameters (x_pos, y_pos, radius)
    circle2 : tuple of floats
        second circle parameters (x_pos, y_pos, radius)

    Returns
    -------
    float
        ratio between area of intersection and area of union

    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    d = math.hypot(x2 - x1, y2 - y1)

    area_intersection = cc_intersection(d, r1, r2)
    area_union = math.pi * (r1 * r1 + r2 * r2) - area_intersection

    return area_intersection / area_union


def cc_intersection(dist, rad1, rad2):
    """
    Area of intersection between two circles

    Parameters
    ----------
    dist : positive float
        distance between circle centers
    rad1 : positive float
        radius of first circle
    rad2 : positive float
        radius of second circle

    Returns
    -------
    intersection_area : positive float
        area of intersection between circles

    References
    ----------
    http://mathworld.wolfram.com/Circle-CircleIntersection.html

    """
    if dist < 0:
        raise ValueError("Distance between circles must be positive")
    if rad1 < 0 or rad2 < 0:
        raise ValueError("Circle radius must be positive")

    if dist == 0 or (dist <= abs(rad2 - rad1)):
        return min(rad1, rad2) ** 2 * math.pi

    if dist > rad1 + rad2 or rad1 == 0 or rad2 == 0:
        return 0

    rad1_sq = rad1 * rad1
    rad2_sq = rad2 * rad2

    circle1 = rad1_sq * math.acos((dist * dist + rad1_sq - rad2_sq) /
                                  (2 * dist * rad1))
    circle2 = rad2_sq * math.acos((dist * dist + rad2_sq - rad1_sq) /
                                  (2 * dist * rad2))
    intersec = 0.5 * math.sqrt((-dist + rad1 + rad2) * (dist + rad1 - rad2) *
                               (dist - rad1 + rad2) * (dist + rad1 + rad2))
    intersection_area = circle1 + circle2 + intersec

    return intersection_area
