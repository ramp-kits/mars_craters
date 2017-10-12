import numpy as np


class ObjectDetector:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        y_pred = [[(1.0, 112.0, 112.0, 112.0)] for img in X]
        y_pred_array = np.empty(len(y_pred), dtype=object)
        y_pred_array[:] = y_pred
        return y_pred_array
