from keras.layers import Conv2D, Lambda
from keras.models import Input, Model
from keras.utils import plot_model
from keras.initializers import he_uniform, glorot_uniform
import keras.backend as K
import tensorflow as tf
import numpy as np


def prepare_pretrained_model(model, crop_index, lock_index, input_shape=(1280, 720, 3), name='FeatureExtractor'):
    """Crops model and locks layers from training
    Receives:
        model - pretrained model
        crop_index - index of last layer in final model
        lock_index - index of last locked layer in final model
        input_shape - since model is convolutional, input could be freely changed
        name - optional model name
    Return:
        new cropped model
    """
    # The only viable option, besides rebuilding the whole model, is to edit model config
    config = model.get_config()
    weights = model.get_weights()

    # Change name
    config['name'] = name

    # Edit input layer
    config['layers'][0]['config']['batch_input_shape'] = (None, *input_shape)

    # Crop unnecessary layers
    config['layers'] = config['layers'][:crop_index + 1]

    # Assign new model output
    config['output_layers'][0][0] = config['layers'][-1]['name']

    # Build cropped model from config and load weights
    model = Model.from_config(config)
    model.set_weights(weights)
    for i in range(lock_index + 1):
        model.layers[i].trainable = False
    return model


def create_rpn_model(model, *, conv_kernels=128, k=9, seed=42):
    """RPN model consists of convolutional layer, which receives feature maps from pretrained model,
    and two heads. Classification head computes a score (how 'signy' (from 0 to 1) insides of respective anchor
    boxes are, in my case), Regression head computes special deltas, which after their application
    on respective anchor box make it better contain object inside
    Receives: feature map
    Outputs: (N, 1) scores
             (N, 4) deltas"""
    conv_layer = Conv2D(
        filters=conv_kernels,
        kernel_size=3,
        padding='same',
        kernel_initializer=he_uniform(seed),
        activation='relu',
        name='RPN_conv'
    )(model.output)
    # Get (fm_width, fm_height, k) tensor
    cls = Conv2D(
        filters=k,
        kernel_size=1,
        kernel_initializer=glorot_uniform(seed),
        activation='sigmoid',
        name='RPN_cls'
    )(conv_layer)
    # Reshape into (-1, 1)
    cls = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 1]),
                 name='bbox_cls')(cls)
    # Get (fm_width, fm_height, k * 4) tensor
    reg = Conv2D(
        filters=4 * k,
        kernel_size=1,
        kernel_initializer=glorot_uniform(seed),
        activation='linear',
        name='RPN_reg'
    )(conv_layer)
    # Reshape into (-1, 4)
    reg = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]),
                 name='bbox_reg')(reg)
    model = Model(inputs=model.input, outputs=[cls, reg], name='RPN')
    # Make losses
    cls_loss = make_cls_processer(tf.keras.losses.binary_crossentropy)
    reg_loss = make_reg_processer(tf.keras.losses.Huber())
    # Make metrics
    cls_acc = make_cls_processer(tf.keras.metrics.binary_accuracy, name='acc')
    reg_mae = make_reg_processer(tf.keras.metrics.mean_absolute_error, name='acc')
    model.compile(optimizer='adadelta',
                  loss={'bbox_cls': cls_loss, 'bbox_reg': reg_loss},
                  metrics={'bbox_cls': cls_acc, 'bbox_reg': reg_mae})
    return model


def make_cls_processer(function, name=None):
    def cls_processer(target_labels, predicted_labels):
        """Ignores anchor boxes labeled as neutral during training
        Receives:
            target_labels: shape (batch_size, N)
            predicted_deltas: shape (batch_size, N, 1)
        Returns:
            function result for all non-neutral samples"""
        # We need to squeeze last dimension in order to not get shape mismatch,
        # even though, length of last dim of predicted_labels is only one.
        # Potential break with 1-sized batch until i figure out how to squeeze along last dimension only
        # tf.squeeze(predicted_labels, -1 (or 2)) doesnt work as shapes somehow could not be
        # received on graph launch
        predicted_labels = tf.squeeze(predicted_labels)
        # Find which targets contribute to the loss (targets with non-neutral labels)
        contributing_indices = tf.where(tf.not_equal(target_labels, -1))
        # Take contributing
        target_labels = tf.gather_nd(target_labels, contributing_indices)
        contributing_prediction = tf.gather_nd(predicted_labels, contributing_indices)
        # Compute loss
        loss = function(target_labels,
                        contributing_prediction)
        # Zero batch size case
        return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    if name is not None:
        cls_processer.__name__ = name
    return cls_processer


def make_reg_processer(function, name=None):
    def reg_processer(targets, predicted_deltas):
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
        loss = function(target_deltas,
                        contributing_prediction)
        # Zero batch size case
        return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    if name is not None:
        reg_processer.__name__ = name
    return reg_processer


def generate_anchor_boxes(sizes, scales, image_size, feature_map_size):
    """Generates all anchor boxes for current RPN configuration.
    Receives:
        sizes: shape (M) - all sizes of 1:1 anchor boxes
        scales: shape (N) - all sides ratios of anchor boxes
        image_size: (iW, iH)
        feature_map_size: (fW, fH)
    Returns:
        anchor_boxes: shape (K, 4) """
    image_width, image_height = image_size
    fm_width, fm_height = feature_map_size
    width_stride = int(image_width / fm_width)
    height_stride = int(image_height / fm_height)

    # Compose horizontal and vertical positions into grid and reshape result into (-1, 2)
    x_centers = np.arange(0, image_width, width_stride)
    y_centers = np.arange(0, image_height, height_stride)
    centers = np.dstack(np.meshgrid(x_centers, y_centers)).reshape((-1, 2))

    # Creates anchor boxes pyramid. Somewhat vectorized version of itertools.product
    r_sides = np.repeat([sizes], len(scales), axis=1).ravel()
    r_scales = np.repeat([scales], len(sizes), axis=0).ravel()
    ab_pyramid = np.transpose([r_sides * (r_scales ** .5),
                               r_sides / (r_scales ** .5)]).astype(int)

    # Creates combinations of all anchor boxes centers and sides
    r_centers = np.repeat(centers, len(ab_pyramid), axis=0)
    r_ab_pyramid = np.repeat([ab_pyramid], len(centers), axis=0).reshape((-1, 2))
    return np.hstack((r_centers, r_ab_pyramid))


def valid_anchor_boxes(anchor_boxes, image_size):
    """Return indices of valid anchor boxes,
    Anchor box is considered valid if it is inside image entirely
    Receives:
        anchor_boxes: shape (N, 4)
        image_size: (iwidth, iheight)
    Returns:
        indices shape (M)
        """
    img_width, img_height = image_size
    x, y, width, height = np.transpose(anchor_boxes)

    # Indicator matrix
    indicators = np.array([x - width // 2 >= 0,
                           y - height // 2 >= 0,
                           x + width // 2 <= img_width,
                           y + height // 2 <= img_height]).transpose()

    # Get indices of anchor boxes inside image
    return np.flatnonzero(np.all(indicators, axis=1, keepdims=False))


def compute_iou(anchor_boxes, gt_boxes):
    """Computes IoU for every anchor box and a batch of ground-truth boxes
    Receives:
        anchor_boxes: shape (N, 4)
        gt_boxes: shape (M, 4)
    Returns:
        ious: shape(N) - max ious for avery anchor box among all gt boxes
        gt_boxes_index: shape(N) - indices of chosen gt boxes"""
    x, y, width, height = np.transpose(anchor_boxes)
    ab_areas = width * height

    # Transform all anchor boxes to 'corners' format
    w_shift = width // 2
    h_shift = height // 2

    ious = []
    for gt_box in gt_boxes:
        x0, y0, x1, y1 = gt_box
        gt_area = (x1 - x0) * (y1 - y0)

        # Compute intersections of all anchor boxes with current gt_box. Result regions could be invalid
        x0 = np.maximum(x0, x - w_shift)
        y0 = np.maximum(y0, y - h_shift)
        x1 = np.minimum(x1, x + w_shift)
        y1 = np.minimum(y1, y + h_shift)

        # Compute intersections area, filtering out any invalid regions
        int_area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)

        ious.append(int_area / (ab_areas + gt_area - int_area))
    # Transpose to make rows represent all ious for every gt box
    # and choose maximum iou with respective gt box index
    ious = np.transpose(ious)
    gt_indices = np.reshape(np.argmax(ious, axis=1), (-1, 1))
    ious = np.squeeze(np.take_along_axis(ious, gt_indices, axis=1))
    # No further need for gt_indices to be 2D array
    return ious, np.squeeze(gt_indices)


def compute_deltas(anchor_boxes, gt_boxes):
    """Computes special deltas between anchor boxes and respective gt boxes
    Receives:
        anchor_boxes [N, x, y, w, h]
        gt_boxes [N, x0, y0, x1, y1]
    Returns:
        deltas [N, (gt_x_center - x) / w,
                   (gt_y_center - y) / h,
                   log(gt_width / w),
                   log(gt_height / h)] - RPN-required deltas"""
    x, y, width, height = np.transpose(anchor_boxes)
    x0, y0, x1, y1 = np.transpose(gt_boxes)

    # Deltas are computed for boxes in 'center' format
    gt_width = x1 - x0
    gt_height = y1 - y0
    gt_x_center = x0 + gt_width // 2
    gt_y_center = y0 + gt_height // 2
    return np.transpose([(gt_x_center - x) / width,
                         (gt_y_center - y) / height,
                         np.log(gt_width / width),
                         np.log(gt_height / height)])


def generate_labels_and_deltas(gt_boxes, anchor_boxes,
                               valid_indices,
                               lower_iou_threshold,
                               upper_iou_threshold,
                               pos_to_neg_ratio,
                               random_generator):
    """Generate RPN training targets - labels and deltas
    Receives:
        gt_boxes [M, x0, y0, x1, y1]
        anchor_boxes [N, x, y, w, h]
        valid_indices [N] - indices of anchor boxes which will be used for computations
        lower_iou_threshold - Any anchor boxes with IoU above this threshold get '1' label (foreground)
        upper_iou_threshold - Any anchor boxes with IoU below this threshold get '1' label (foreground)
        random_generator - np random generator
    Returns:
        labels [N]
        deltas [N, dx, dy, dw, dh]"""
    ious = np.zeros(anchor_boxes.shape[0])

    # Compute IoU for valid anchor boxes and get indices of respective gt boxes
    ious[valid_indices], gt_boxes_indices = compute_iou(anchor_boxes[valid_indices], gt_boxes)
    labels = np.full(ious.shape, -1)

    # Positive samples. We find indices of positive samples in array of all anchor boxes
    positive_indices = np.flatnonzero((ious > upper_iou_threshold)[valid_indices])
    labels[positive_indices] = 1

    # Indices of respective gt_boxes in array of valid anchor boxes
    gt_boxes_indices = gt_boxes_indices[ious[valid_indices] > upper_iou_threshold]

    # Negative samples
    max_negative_indices = int(len(positive_indices) / pos_to_neg_ratio)
    negative_indices = np.flatnonzero((ious < lower_iou_threshold)[valid_indices])
    if len(negative_indices) > max_negative_indices:
        negative_indices = random_generator.choice(negative_indices,
                                                   max_negative_indices,
                                                   replace=False)
    labels[negative_indices] = 0

    # Find positions of positive gt boxes
    gt_boxes = np.take(gt_boxes, gt_boxes_indices, axis=0)
    deltas = np.zeros_like(anchor_boxes, dtype='float')
    deltas[positive_indices] = compute_deltas(anchor_boxes[positive_indices], gt_boxes)

    # Labels are 1D, deltas are a 2D, so we should expand labels dimensions
    # Intentional, as loss function without lambdas (which are not usable in our
    # case as we have to save and load models) receives only one tensor
    return np.hstack((labels[:, np.newaxis], deltas))


def prepare_rpn_anchor_boxes(image_size, feature_map_size, sizes, scales):
    """Generate anchor boxes and indices of valid anchor boxes beforehand
    Receives:
        image_size (iwidth, iheight)
        feature_map_size (fwidth, fheight)
        sizes - array of 1:1 anchor box sides
        scales - array of width to height ratios for anchor boxes
    Returns:
        anchor_boxes [N [x, y, w, h]] - all image anchor boxes
        valid_ab_indices [M, index] - indices of valid anchor boxes"""
    anchor_boxes = generate_anchor_boxes(sizes, scales, image_size, feature_map_size)
    valid_ab_indices = valid_anchor_boxes(anchor_boxes, image_size)
    return anchor_boxes, valid_ab_indices


def rpn_generator(data_generator, *,
                  anchor_boxes,
                  valid_ab_indices,
                  lower_iou_threshold,
                  upper_iou_threshold,
                  pos_to_neg_ratio=0.5,
                  seed=42
    ):
    """Generates RPN targets - feature maps and anchor boxes labels & deltas - in a traditional way
    Receives:
        data_generator - generator, returning feature maps and ground-truth boxes in batches
        anchor_boxes [N, [x, y, w, h]] - all image anchor_boxes
        valid_ab_indices [M, index] - indices of valid anchor_boxes
        lower_iou_threshold - zero to one float, anchor_boxes with gt iou > this threshold
                              will be labeled as positive ('1'/ 'foreground')
        lower_iou_threshold - zero to one float, anchor_boxes with gt iou < this threshold
                              will be labeled as negative ('0'/ 'background')
        pos_to_neg_ratio - float, defines how much negative samples will be generated according
                           to a number of positive samples.
                           F.e: ptnr = 0.5,
                                len(pos_samples) = 128
                                len(negative_samples) = 128 / 0.5 = 256
        seed - generator seed. Generator is isolated, so it will always give defined sequence of data
    Returns:
        spits out batches of feature maps and rpn targets in format
            [batch_size, feature_maps],
            [[batch_size, labels], [batch_size, [label, dx, dy, dw, dh]]"""
    random_generator = np.random.default_rng(seed=seed)
    for imgs_batch, gt_boxes_batch in data_generator:
        # Generate bbox_reg targets
        targets = np.array([generate_labels_and_deltas(gt_boxes,
                                                       anchor_boxes,
                                                       valid_ab_indices,
                                                       lower_iou_threshold,
                                                       upper_iou_threshold,
                                                       pos_to_neg_ratio,
                                                       random_generator)
                            for gt_boxes in gt_boxes_batch])
        # Intentionally squeezes last dimension
        labels = targets[:, :, 0]
        yield imgs_batch, [labels, targets]