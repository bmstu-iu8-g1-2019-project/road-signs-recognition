import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Activation, Reshape, Lambda, Input
from keras.initializers import he_uniform, glorot_uniform
from keras.losses import SparseCategoricalCrossentropy, Huber
from keras.metrics import SparseCategoricalAccuracy, mean_absolute_error
from layers import ApplyDeltas, NonMaximumSuppression, RegionOfInterestPooling


def prepare_rpn_model(model,
                      anchor_boxes,
                      valid_indices,
                      overlap_th,
                      rois_n,
                      image_size=(640, 360),
                      target_size=(7, 7)):
    # Applies deltas and transforms regions into 'corners' format
    regions = ApplyDeltas(anchor_boxes, valid_indices)([model.output[2], model.output[1]])
    # Discard most of the regions. Returns ragged tensor
    regions = NonMaximumSuppression(overlap_th, rois_n)(regions)
    # Apply ROIP to make all feature maps have equal shapes. Returns ragged tensor
    roip = RegionOfInterestPooling(image_size, target_size)([model.output[3], regions])
    new_model = Model(inputs=[model.input], outputs=[roip, regions])
    return new_model


def create_detector_model(class_n, roip_fm_shape, class_indices_dict, seed=42):
    he_init = he_uniform(seed)
    gl_init = glorot_uniform(seed)
    input = Input(roip_fm_shape)
    output = Flatten(name='DetectorFlatten')(input)
    output = Dense(
        units=500,
        kernel_initializer=he_init,
        activation='relu',
        name='DetectorDense1')(output)
    output = Dense(
        units=500,
        kernel_initializer=he_init,
        activation='relu',
        name='DetectorDense2')(output)
    cls_logits = Dense(
        units=class_n + 1,  # Add background class
        kernel_initializer=gl_init,
        activation='linear',
        name='DetectorClsLogits')(output)
    cls = Activation('softmax', name='DetectorCls')
    reg = Dense(
        units=4 * class_n,
        kernel_initializer=he_init,
        activation='linear',
        name='DetectorReg'
    )(output)
    reg = Reshape(target_shape=(-1, 4))(reg)
    model = Model(inputs=[input], outputs=[cls_logits, cls, reg])
    # Since only deltas from respective class contribute to the loss
    reg_loss = on_index_wrapper(Huber(), class_indices_dict['not_sign'])
    # Computes metric on batch - Keras can`t reset metric on batch end if it is wrapped
    reg_metric = on_index_wrapper(mean_absolute_error, class_indices_dict['not_sign'])
    model.compile(optimizer='adadelta',
                  loss={'DetectorClsLogits': SparseCategoricalCrossentropy(),
                        'DetectorCls': None,
                        'DetectorReg': reg_loss},
                  metrics={'DetectorCls': SparseCategoricalAccuracy(),
                           'DetectorReg': reg_metric})
    return model


def class_indices(classes_arr):
    """Returns a dict with classes mapped to indices"""
    size = len(classes_arr)
    return {classes_arr[i]: i for i in range(size)}


def on_index_wrapper(function, class_indices_dict, background_label, name=None):
    classes_number = len(class_indices_dict)
    background_index = tf.cast(class_indices_dict[background_label], dtype=tf.int32)

    def f(y_true, x_pred):
        # Extract indices and labels
        class_indices = y_true[:, classes_number]
        true_labels = y_true[:, classes_number:]
        # Filter out samples with background label
        non_bg_mask = tf.not_equal(class_indices[:, background_label], background_index)
        class_indices = tf.boolean_mask(class_indices, non_bg_mask)
        true_labels = tf.boolean_mask(true_labels, non_bg_mask)
        # Gather respective indices from every sample and squeeze 1 dimension
        # TODO(Mocurin)
        pred_labels = tf.squeeze(tf.gather(x_pred,
                                           class_indices[:, tf.newaxis],
                                           axis=1, batch_dims=1))
        # Since class_indices were 2D, we need to squeeze first axis of gathered tensor
        res = function(true_labels,
                       pred_labels)
        return K.switch(tf.size(res) > 0, res, tf.constant(0.0))
    # If function is used as metric its 'label' during logging is received from function.__name__
    if name is not None:
        f.__name__ = name
    return f


if __name__ == '__main__':
    tf.executing_eagerly()
    classl = tf.constant([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])
    labels = tf.constant([[[0, 1], [0, 2], [0, 3], [0, 4]],
                          [[0, 1], [0, 2], [0, 3], [0, 4]],
                          [[0, 1], [0, 2], [0, 3], [0, 4]],
                          [[0, 1], [0, 2], [0, 3], [0, 4]]])
    print(tf.gather_nd(labels, classl))
