import numpy as np

from joblib import Parallel, delayed

from sklearn.base import clone
from imblearn.ensemble import BalancedBaggingClassifier

from .extraction import BlobExtractor


class ObjectDetector(object):
    def __init__(self, extractor=None, estimator=None, n_jobs=1):
        self.extractor = extractor
        self.estimator = estimator
        self.n_jobs = n_jobs

    def _extract_features(self, X, y):
        # extract feature for all the image containing craters
        data_extracted = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extractor_.fit_extract)(image, craters)
            for image, craters in zip(X, y))

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
            self.estimator_ = BalancedBaggingClassifier(n_jobs=self.n_jobs)
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
            output[img_idx].append((crater[0], crater[1], crater[2],
                                    pred[crater_idx]))

        return output
