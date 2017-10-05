from IPython.display import display

import numpy as np

from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter


class ObjectDetector:
    def __init__(self, sigma=3, threshold=0.4):
        self.sigma = sigma
        self.threshold = threshold

    def fit(self, gen_builder):
        return self

    def _hough_detection(self, image):
        edges = canny(image, sigma=self.sigma, low_threshold=10,
                      high_threshold=50)

        hough_radii = list(np.arange(5, 20, 1)) + list(np.arange(20, 50, 2))

        circles = hough_circle(edges, hough_radii)
        # print('Max val: ', np.max(circles))
        peaks = hough_circle_peaks(circles, hough_radii,
                                   threshold=self.threshold)
        return edges, peaks

    def predict(self, X):
        return [self._predict_single(img) for img in X]

    def _predict_single(self, X):
        edges, peaks = self._hough_detection(X)
        accum, cx, cy, radii = peaks
        return list(zip(cx, cy, radii, accum))

    def show_prediction(self, X):
        import matplotlib.pyplot as plt

        edges, peaks = self._hough_detection(X)
        accum, cx, cy, radii = peaks

        image = X

        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(15, 6))
        image2 = color.gray2rgb(image)
        _, cx, cy, radii = peaks
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius)
            image2[circy, circx] = (220, 20, 20)

        print("Radius of circles", peaks[3])

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[1].imshow(edges, cmap=plt.cm.gray)
        ax[2].imshow(image2, cmap=plt.cm.gray)
        display(fig)
