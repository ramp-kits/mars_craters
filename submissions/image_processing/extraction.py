from itertools import repeat

import numpy as np

from sklearn.base import BaseEstimator
from skimage.feature import blob_doh
from mahotas.features import zernike_moments
from mahotas.features import surf

from .iou import cc_iou


class ExtractorMixin(object):
    """Mixin class for feature extraction to modify initial X and y."""
    _estimator_type = 'extractor'

    def fit_extract(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).extract(X)
        else:
            return self.fit(X, y, **fit_params).extract(X, y)


class BlobExtractor(BaseEstimator, ExtractorMixin):

    def __init__(self, min_radius=5, max_radius=40, blob_threshold=0.01,
                 overlap=0.5, padding=1.2, iou_threshold=0.5):
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
        x_min = x - padded_radius
        x_min = x_min if x_min < 0 else 0
        y_min = y - padded_radius
        y_min = y_min if y_min < 0 else 0
        x_max = x + padded_radius
        x_max = x_max if x_max > X.shape[0] else X.shape[0] - 1
        y_max = y + padded_radius
        y_max = y_max if y_max > X.shape[1] else X.shape[1] - 1

        patch = X[y_min:y_max, x_min:x_max]

        # compute Zernike moments
        zernike = zernike_moments(patch, radius)

        # compute SURF descriptor
        keypoint = np.array([[y, x, 1, 0.1, 1]])
        surf_descriptor = surf.descriptors(patch, keypoint,
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
