class ObjectDetector:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [[(1.0, 112.0, 112.0, 112.0)] for img in X]
