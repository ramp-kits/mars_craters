from __future__ import division

import imp
import numpy as np


class ObjectDetector(object):
    """
    Object detection workflow.

    This workflow is used to train image object detection tasks, typically
    when the dataset cannot be stored in memory.
    Submissions need to contain two files, which by default are named:
    image_preprocessor.py and object_detector_model.py (they can be
    modified by changing `workflow_element_names`).

    image_preprocessor.py needs a `tranform` function, which
    is used for preprocessing the images. It takes an image as input
    and it returns an image as an output. Optionally, image_preprocessor.py
    can also have a function `transform_test`, which is used only to preprocess
    images at test time. Otherwise, if `transform_test` does not exist,
    `transform` is used at train and test time.

    object_detector_model.py needs a `ObjectDetector` class, which
    implements `fit` and `predict`.

    Parameters
    ==========

    test_batch_size : int
        batch size used for testing.

    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).

    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.

    """
    def __init__(self, test_batch_size, chunk_size, n_jobs,
                 workflow_element_names=[
                     'image_preprocessor', 'object_detector_model']):
        self.element_names = workflow_element_names
        self.test_batch_size = test_batch_size
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs

    def train_submission(self, module_path, X, y, train_is=None):
        """Train a ObjectDetector.

        module_path : str
            module where the submission is. the folder of the module
            have to contain batch_classifier.py and image_preprocessor.py.
        X_array : ArrayContainer vector of int
            vector of image IDs to train on
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        y_array : vector of int
            vector of image labels corresponding to X_train
        train_is : vector of int
           indices from X_array to train on
        """
        if train_is is None:
            train_is = slice(None, None, None)

        # image transformation preprocessing
        submitted_image_preprocessor_file = '{}/{}.py'.format(
            module_path, self.element_names[0])
        image_preprocessor = imp.load_source(
            '', submitted_image_preprocessor_file)
        transform_img = image_preprocessor.transform
        transform_test_img = getattr(image_preprocessor,
                                     'transform_test',
                                     transform_img)

        # object detector model
        submitted_model_file = '{}/{}.py'.format(
            module_path, self.element_names[1])
        detector = imp.load_source('', submitted_model_file)
        clf = detector.ObjectDetector()

        # train and return fitted model
        gen_builder = BatchGeneratorBuilder(
            X[train_is], y[train_is], transform_img,
            chunk_size=self.chunk_size, n_jobs=self.n_jobs)
        clf.fit(gen_builder)
        return transform_img, transform_test_img, clf

    def test_submission(self, trained_model, X):
        """Test an ObjectDetector.

        trained_model : tuple
            tuple of a trained model returned by `train_submission`.
        X_array : ArrayContainer of int
            vector of image IDs to test on.
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        """
        transform_img, transform_test_img, clf = trained_model
        it = _chunk_iterator(X, chunk_size=self.chunk_size)
        y_pred = []
        for X in it:
            for i in range(0, len(X), self.test_batch_size):
                # 1) Preprocessing
                X_batch = X[i: i + self.test_batch_size]
                # X_batch = Parallel(n_jobs=self.n_jobs, backend='threading')(
                #     delayed(transform_img)(x) for x in X_batch)
                X_batch = [transform_test_img(x) for x in X_batch]
                # X is a list of numpy arrrays at this point, convert it to a
                # single numpy array.
                # try:
                #     X_batch = [x[np.newaxis, :, :, :] for x in X_batch]
                # except IndexError:
                #     # single channel
                #     X_batch = [
                #         x[np.newaxis, np.newaxis, :, :] for x in X_batch]
                # X_batch = np.concatenate(X_batch, axis=0)

                # 2) Prediction
                y_pred_batch = clf.predict(X_batch)
                y_pred.extend(y_pred_batch)

        # y_pred = np.concatenate(y_pred, axis=0)
        return y_pred


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).
    An instance of this class is exposed to users `Classifier` through
    the `fit` function : model fitting is called by using
    "clf.fit(gen_builder)" where `gen_builder` is an instance
    of this class : `BatchGeneratorBuilder`.
    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int
        vector of image labels corresponding to `X_array`

    folder : str
        folder where the images are

    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).

    n_classes : int
        Total number of classes. This is needed because the array
        of labels, which is a vector of ints, is transformed into
        a onehot representation.

    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.
    """

    def __init__(self, X_array, y_array, transform_img, chunk_size, n_jobs):
        self.X_array = X_array
        self.y_array = y_array
        self.transform_img = transform_img
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.nb_examples = len(X_array)

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            it = _chunk_iterator(
                X_array=self.X_array[indices], y_array=self.y_array[indices],
                chunk_size=self.chunk_size)
            for X, y in it:
                # 1) Preprocessing of X and y
                # X = Parallel(n_jobs=self.n_jobs, backend='threading')(
                    # delayed(self.transform_img)(x) for x in X)
                X = np.array([self.transform_img(x) for x in X])
                # # X is a list of numpy arrrays at this point, convert it to a
                # single numpy array.
                try:
                    X = [x[np.newaxis, :, :, :] for x in X]
                except IndexError:
                    # single channel
                    X = [x[np.newaxis, np.newaxis, :, :] for x in X]
                X = np.concatenate(X, axis=0)
                X = np.array(X, dtype='float32')
                # Convert y to onehot representation
                # y = _to_categorical(y, num_classes=self.n_classes)

                # 2) Yielding mini-batches
                for i in range(0, len(X), batch_size):
                    yield X[i:i + batch_size], y[i:i + batch_size]


def _chunk_iterator(X_array, y_array=None, chunk_size=1024):
    """Generate chunks of images, optionally with their labels.

    Parameters
    ==========

    X_array : ArrayContainer of int
        image ids to load
        (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int
        labels corresponding to each image from X_array

    chunk_size : int
        chunk size

    Yields
    ======

    if y_array is provided (not None):
        it yields each time a tuple (X, y) where X is a list
        of numpy arrays of images and y is a list of ints (labels).
        The length of X and y is `chunk_size` at most (it can be smaller).

    if y_array is not provided (it is None)
        it yields each time X where X is a list of numpy arrays
        of images. The length of X is `chunk_size` at most (it can be
        smaller).
        This is used for testing, where we don't have/need the labels.

    """
    for i in range(0, X_array.shape[0], chunk_size):
        X = X_array[i:i + chunk_size, :, :]

        if y_array is not None:
            y = y_array[i:i + chunk_size]
            yield X, y
        else:
            yield X
