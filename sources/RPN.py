from keras.layers import Conv2D, Lambda
from keras.models import Input, Model
from keras.losses import binary_crossentropy, huber_loss
import keras.backend as K
import tensorflow as tf


def create_rpn_model(input_shape=(None, None, None),
                     ab_per_fm_point=9):
    feature_map = Input(shape=input_shape)
    conv_layer = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same'
    )(feature_map)
    cls = Conv2D(
        filters=ab_per_fm_point,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="RPN_cls"
    )(conv_layer)
    # Ravel [batch, width, height, ab_scores] into [batch, ab_scores]
    cls = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 1]))(cls)
    reg = Conv2D(
        filters=4 * ab_per_fm_point,
        kernel_size=(1, 1),
        activation="linear",
        kernel_initializer="uniform",
        name="RPN_reg"
    )(conv_layer)
    # Ravel [batch, width, height, ab_deltas] into [batch, ab_deltas]
    reg = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]))(reg)
    model = Model(inputs=[feature_map], outputs=[cls, reg])
    model.compile(optimizer='adam', loss={'RPN_cls': binary_crossentropy, 'RPN_reg': huber_loss})
    return model


def cls_loss(target_labels, predicted_label):
    """Wrapper for a binary crossentropy loss of bounding box regression.
    Makes that neutral samples do not contribute to the loss"""
    target_labels = tf.squeeze(target_labels, -1)
    contributing_indices = tf.where(tf.not_equal(target_labels, -1))
    target_labels = tf.gather_nd(target_labels, contributing_indices)
    contributing_prediction = tf.gather_nd(predicted_label, contributing_indices)
    loss = K.binary_crossentropy(target=target_labels,
                                 output=contributing_prediction)
    return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def bbox_loss(target_deltas, target_labels, predicted_deltas):
    """Wrapper for a loss of anchor box classification.
    Makes that negative and neutral samples do not contribute to the loss"""
    target_labels = tf.squeeze(target_labels, -1)
    contributing_indices = tf.where(tf.equal(target_labels, 1))
    target_deltas = tf.gather_nd(target_deltas, contributing_indices)
    contributing_prediction = tf.gather_nd(predicted_deltas, contributing_indices)
    loss = huber_loss(target_deltas,
                      contributing_prediction)
    return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def rpn_generator(
        image_data_generator,            # any data generator which outputs image data and ground-truth box position
        pretrained_model, *,             # model, which will be used with rpn
        lower_iou_threshold=0.1,         # threshold for BG label
        upper_iou_threshold=0.5,         # threshold for FG label
        ab_sizes=(32, 64, 96),           # side of 1:1 anchor box
        ab_scales=(0.5, 1., 2.),         # width to height anchor box ratio
        positive_to_negative_ratio=0.5,  # FG to BG ratio in a batch
        fill_samples='neutral',          # label that is used to fill batch after threshold application
        batch_size=256,                  # size of generated batches (number of FG + BG samples
        seed=42                          # seed for BG samples generation
    ):
    """Generates data for RPN training. Takes a generator (Keras ImageDataGenerator preferably)
    and a model which will be used with RPN. Outputs a batch (two tensors - label && position) of training data"""

    pass

