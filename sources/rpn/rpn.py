from keras.layers import Conv2D, Activation, Reshape
from keras.models import Model
from keras.initializers import he_uniform, glorot_uniform
from keras.regularizers import Regularizer
import tensorflow as tf
import keras.backend as K
import numpy as np


def training_rpn_model(feature_extractor, anchors_per_loc, seed):
    conv_init = he_uniform(seed)
    cls_reg_init = glorot_uniform(seed)
    reg_3x3 = ThresholdedRegularizer(penalty=0.1, threshold=0.5, kernel_shape=(3, 3, 64))
    reg_1x1 = ThresholdedRegularizer(penalty=0.1, threshold=0.1)

    conv = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=conv_init,
        kernel_regularizer=reg_3x3,
        activation='relu',
        name='RPN_conv'
    )(feature_extractor.output)
    cls = Conv2D(
        filters=1 * anchors_per_loc,
        kernel_size=(1, 1),
        kernel_initializer=cls_reg_init,
        kernel_regularizer=reg_1x1,
        activation='sigmoid',
        name='RPN_cls'
    )(conv)
    reg = Conv2D(
        filters=4 * anchors_per_loc,
        kernel_size=(1, 1),
        kernel_initializer=cls_reg_init,
        kernel_regularizer=reg_1x1,
        activation=expanded_sigmoid,
        name='RPN_reg'
    )(conv)
    cls = Reshape(target_shape=(-1,), name='bbox_cls')(cls)
    reg = Reshape(target_shape=(-1, 4), name='bbox_reg')(reg)
    return Model(inputs=feature_extractor.input, outputs=[cls, reg, feature_extractor.output], name='RPN')


def clean_rpn_model(feature_extractor, anchors_per_loc):
    conv = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        name='RPN_conv'
    )(feature_extractor.output)
    cls = Conv2D(
        filters=1 * anchors_per_loc,
        kernel_size=(1, 1),
        activation='sigmoid',
        name='RPN_cls'
    )(conv)
    reg = Conv2D(
        filters=4 * anchors_per_loc,
        kernel_size=(1, 1),
        activation=expanded_sigmoid,
        name='RPN_reg'
    )(conv)
    cls = Reshape(target_shape=(-1,), name='bbox_cls')(cls)
    reg = Reshape(target_shape=(-1, 4), name='bbox_reg')(reg)
    return Model(inputs=feature_extractor.input, outputs=[cls, reg, feature_extractor.output], name='RPN')


class ThresholdedRegularizer(Regularizer):
    def __init__(self, penalty=0.1, threshold=0.1, kernel_shape=None):
        self.penalty = K.cast_to_floatx(penalty)
        if kernel_shape is None:
            self.threshold = K.cast_to_floatx(threshold)
        else:
            self.threshold = K.cast_to_floatx(threshold * np.sqrt(np.prod(kernel_shape)))

    def __call__(self, x):
        def l2(w):
            return K.sum(K.square(w))
        return self.penalty * tf.reduce_sum(tf.cast(tf.map_fn(l2, x) < self.threshold, dtype=tf.float32))

    def get_config(self):
        return {'penalty': float(self.penalty), 'threshold': float(self.threshold)}


def make_cls_wrapper(function, name=None):
    def cls_wrapper(target_labels, predicted_labels):
        """Ignores anchor boxes labeled as neutral during training
        Receives:
            target_labels: shape (batch_size, N)
            predicted_deltas: shape (batch_size, N)
        Returns:
            function result for all non-neutral samples"""
        # Find which targets contribute to the loss (targets with non-neutral labels)
        contributing_indices = tf.where(tf.not_equal(target_labels, -1))

        # Take contributing
        target_labels = tf.gather_nd(target_labels, contributing_indices)
        contributing_prediction = tf.gather_nd(predicted_labels, contributing_indices)

        # Compute loss
        res = function(target_labels,
                       contributing_prediction)

        # Zero batch size case
        return K.switch(tf.size(res) > 0, K.mean(res), tf.constant(0.0))
    if name is not None:
        cls_wrapper.__name__ = name
    return cls_wrapper


def make_reg_wrapper(function, name=None):
    def reg_wrapper(targets, predicted_deltas):
        """Ignores anchor boxes labeled as negative or neutral
        Receives:
            targets: shape (batch_size, N, 5) - tensor, which contains labels at [:, :, 0]
                                                and deltas at [:, :, 1:5]
            predicted_deltas: shape (batch_size, N, 4)
        Returns:
            function result for all positive samples"""
        # Extract labels, intentionally squeezed
        target_labels = targets[:, :, 0]

        # Extract deltas
        target_deltas = targets[:, :, 1:]

        # Find which targets contribute to the loss (targets with positive labels)
        contributing_indices = tf.where(tf.equal(target_labels, 1))

        # Take contributing
        target_deltas = tf.gather_nd(target_deltas, contributing_indices)
        contributing_prediction = tf.gather_nd(predicted_deltas, contributing_indices)

        # Compute loss
        res = function(target_deltas,
                       contributing_prediction)
        # Zero batch size case
        return K.switch(tf.size(res) > 0, K.mean(res), tf.constant(0.0))
    if name is not None:
        reg_wrapper.__name__ = name
    return reg_wrapper


class ClsMetricWrapper:
    """Most metrics need to be reset on batch end, so i`ve created simple workaround
    for keras to think this is real metric. Could be done through an inheritance
    for a particular metric"""
    def __init__(self, metric, name=None):
        self.stateful = True
        # self.built = True
        self.metric = metric
        self.metric_wrapper = make_cls_wrapper(metric, name)
        self.__name__ = 'cls_metric'
        if name is not None:
            self.__name__ = name

    def __call__(self, y_true, x_pred):
        return self.metric_wrapper(y_true, x_pred)

    def reset_states(self):
        self.metric.reset_states()


class RegMetricWrapper:
    """Most metrics need to be reset on batch end, so i`ve created simple workaround
    for keras to think this is real metric. Could be done through an inheritance
    for a particular metric"""
    def __init__(self, metric, name=None):
        self.stateful = True
        # self.built = True
        self.metric = metric
        self.metric_wrapper = make_reg_wrapper(metric, name)
        self.__name__ = 'reg_metric'
        if name is not None:
            self.__name__ = name

    def __call__(self, y_true, x_pred):
        return self.metric_wrapper(y_true, x_pred)

    def reset_states(self):
        self.metric.reset_states()


def expanded_sigmoid(x):
    return -1 + 2 * K.sigmoid(x)
