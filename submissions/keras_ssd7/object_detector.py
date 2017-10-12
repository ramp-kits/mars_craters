from __future__ import division

from math import ceil

import numpy as np

from sklearn.utils import Bunch

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from ssd_keras.keras_ssd7 import build_model
from ssd_keras.keras_ssd_loss import SSDLoss
from ssd_keras.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y


class ObjectDetector(object):
    """Object detector.

    Parameters
    ----------
    batch_size : int, optional
        The batch size used during training. Set by default to 32 samples.

    epoch : int, optional
        The number of epoch for which the model will be trained. Set by default
        to 50 epochs.

    model_check_point : bool, optional
        Whether to create a callback for intermediate models.

    Attributes
    ----------
    model_ : object
        The SSD keras model.

    params_model_ : Bunch dictionary
        All hyper-parameters to build the SSD model.

    """

    def __init__(self, batch_size=32, epoch=50, model_check_point=True):
        self.model_, self.params_model_, self.predictor_sizes = \
            self._build_model()
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_check_point = model_check_point

    def fit(self, X, y):
        
        ### TEMP - for showcase load weights (this is not possible
        # for an actual submission)
        self.model_.load_weights('submissions/ssd7/ssd7_0_weights.h5')
        return
        ###
        
        # build the box encoder to later encode y to make usable in the model
        ssd_box_encoder = SSDBoxEncoder(
            img_height=self.params_model_.img_height,
            img_width=self.params_model_.img_width,
            n_classes=self.params_model_.n_classes,
            predictor_sizes=self.predictor_sizes,
            min_scale=self.params_model_.min_scale,
            max_scale=self.params_model_.max_scale,
            scales=self.params_model_.scales,
            aspect_ratios_global=self.params_model_.aspect_ratios,
            two_boxes_for_ar1=self.params_model_.two_boxes_for_ar1,
            pos_iou_threshold=0.5,
            neg_iou_threshold=0.2)

        train_dataset = BatchGeneratorBuilder(X, y, ssd_box_encoder)
        train_generator, val_generator, n_train_samples, n_val_samples = \
            train_dataset.get_train_valid_generators(
                batch_size=self.batch_size)

        # create the callbacks to get during fitting
        callbacks = []
        if self.model_check_point:
            callbacks.append(
                ModelCheckpoint('./ssd7_0_weights_epoch{epoch:02d}_'
                                'loss{loss:.4f}.h5',
                                monitor='val_loss', verbose=1,
                                save_best_only=True, save_weights_only=True,
                                mode='auto', period=1))
        # add early stopping
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
                                       patience=5))

        # reduce learning-rate when reaching plateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                           patience=0, epsilon=0.001,
                                           cooldown=0))

        # fit the model
        self.model_.fit_generator(
            generator=train_generator,
            steps_per_epoch=ceil(n_train_samples / self.batch_size),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=ceil(n_val_samples / self.batch_size))

    @staticmethod
    def _anchor_to_circle(boxes, pred=True):
        """Convert the anchor boxes predicted to circlular predictions.

        Parameters
        ----------
        boxes : list of tuples
            Each tuple is organized as [confidence, x_min, x_max, y_min, y_max]

        pred : bool, default=True
            Set to True if boxes represent some predicted anchor boxes (i.e.,
            contains some prediction confidence)

        Returns
        -------
        circles : list of tuples
            Each tuple is organized as [confidence, cy, cx, radius] if pred is
            True or [cy, cx, radius] otherwise.

        """
        res = []
        for box in boxes:
            if pred:
                box = box[1:]
            conf, x_min, x_max, y_min, y_max = box
            radius = (((x_max - x_min) + (y_max - y_min)) / 2) / 2
            cx = x_min + (x_max - x_min) / 2
            cy = y_min + (y_max - y_min) / 2
            if pred:
                res.append((conf, cy, cx, radius))
            else:
                res.append((cy, cx, radius))
        return res

    def predict(self, X):
        y_pred = self.model_.predict(np.expand_dims(X, -1))
        # only the 15 best candidate will be kept
        y_pred_decoded = decode_y(y_pred, top_k=15, input_coords='centroids')
        return np.array([self._anchor_to_circle(x, pred=True)
                         for x in y_pred_decoded])

    ###########################################################################
    # Setup SSD model

    @staticmethod
    def _init_params_model():
        params_model = Bunch()

        # image and class parameters
        params_model.img_height = 224
        params_model.img_width = 224
        params_model.img_channels = 1
        params_model.n_classes = 2

        # window detection parameters
        params_model.min_scale = 0.08
        params_model.max_scale = 0.96
        params_model.scales = [0.08, 0.16, 0.32, 0.64, 0.96]
        params_model.aspect_ratios = [1.0]
        params_model.two_boxes_for_ar1 = False

        # optimizer parameters
        params_model.lr = 0.001
        params_model.beta_1 = 0.9
        params_model.beta_2 = 0.999
        params_model.epsilon = 1e-08
        params_model.decay = 5e-05

        # loss parameters
        params_model.neg_pos_ratio = 3
        params_model.n_neg_min = 0
        params_model.alpha = 1.0

        return params_model

    def _build_model(self):

        # load the parameter for the SSD model
        params_model = self._init_params_model()

        model, predictor_sizes = build_model(
            image_size=(params_model.img_height,
                        params_model.img_width,
                        params_model.img_channels),
            n_classes=params_model.n_classes,
            min_scale=params_model.min_scale,
            max_scale=params_model.max_scale,
            scales=params_model.scales,
            aspect_ratios_global=params_model.aspect_ratios,
            two_boxes_for_ar1=params_model.two_boxes_for_ar1)

        adam = Adam(lr=params_model.lr, beta_1=params_model.beta_1,
                    beta_2=params_model.beta_2, epsilon=params_model.epsilon,
                    decay=params_model.decay)

        ssd_loss = SSDLoss(neg_pos_ratio=params_model.neg_pos_ratio,
                           n_neg_min=params_model.n_neg_min,
                           alpha=params_model.alpha)

        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return model, params_model, predictor_sizes

###############################################################################
# Batch generator


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """

    def __init__(self, X_array, y_array, ssd_box_encoder):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)
        self.ssd_box_encoder = ssd_box_encoder

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
            X = self.X_array[indices]
            y = [self.y_array[i] for i in indices]

            # converting to float needed?
            # X = np.array(X, dtype='float32')

            # Yielding mini-batches
            for i in range(0, len(X), batch_size):
                X_batch = [np.expand_dims(img, -1)
                           for img in X[i:i + batch_size]]
                y_batch = y[i:i + batch_size]

                y_batch = [np.array([(1, cx - r, cy - r, cx + r, cy + r)
                                     for (cy, cx, r) in y_patch])
                           for y_patch in y_batch]

                y_batch_encoded = self.ssd_box_encoder.encode_y(y_batch)

                yield np.array(X_batch), y_batch_encoded
