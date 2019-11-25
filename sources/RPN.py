from keras.layers import Conv2D, Lambda
from keras.models import Input, Model
from keras.losses import binary_crossentropy, huber_loss
import keras.backend as K
import tensorflow as tf
import numpy as np


def create_rpn_model(input_shape, ab_per_fm_point=9):
    """Model architecture"""
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
    # Reshape into (-1, 1)
    cls = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 1]))(cls)
    reg = Conv2D(
        filters=4 * ab_per_fm_point,
        kernel_size=(1, 1),
        activation="linear",
        kernel_initializer="uniform",
        name="RPN_reg"
    )(conv_layer)
    # Reshape into (-1, 4)
    reg = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]))(reg)
    model = Model(inputs=[feature_map], outputs=[cls, reg])
    model.compile(optimizer='adadelta', loss={'RPN_cls': cls_loss, 'RPN_reg': bbox_loss})
    return model


def cls_loss(target_labels, predicted_label):
    """binary_crossentropy loss wrapper. Ignores anchor boxes labeled as neutral
    during training"""
    target_labels = tf.squeeze(target_labels, -1)
    contributing_indices = tf.where(tf.not_equal(target_labels, -1))
    target_labels = tf.gather_nd(target_labels, contributing_indices)
    contributing_prediction = tf.gather_nd(predicted_label, contributing_indices)
    loss = K.binary_crossentropy(target=target_labels,
                                 output=contributing_prediction)
    # Zero batch size case
    return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def bbox_loss(target_deltas, target_labels, predicted_deltas):
    """huber_loss wrapper. Ignores anchor boxes labeled as negative or neutral"""
    target_labels = tf.squeeze(target_labels, -1)
    contributing_indices = tf.where(tf.equal(target_labels, 1))
    target_deltas = tf.gather_nd(target_deltas, contributing_indices)
    contributing_prediction = tf.gather_nd(predicted_deltas, contributing_indices)
    loss = huber_loss(target_deltas,
                      contributing_prediction)
    # Zero batch size case
    return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def generate_anchor_boxes(sizes, scales, image_size, feature_map_size):
    """Generates all anchor boxes for current RPN configuration.
    Receives:
        sizes [M] - all sizes of 1:1 anchor boxes
        scales [S] - all sides ratios of anchor boxes
        image_size (W, H)
        feature_map_size (fW, fH)

    Returns:
        all_ab [N, x, y, w, h]"""
    image_width, image_height = image_size
    width_stride, height_stride = np.floor(np.divide(image_size, feature_map_size)).astype(int)

    # Compose horizontal and vertical positions into grid and reshape result into (-1, 2)
    x_centers = np.arange(0, image_width, width_stride)
    y_centers = np.arange(0, image_height, height_stride)
    centers = np.dstack(np.meshgrid(x_centers, y_centers)).reshape((-1, 2))

    # Creates anchor boxes pyramid. Somewhat vectorized version of itertools.product
    r_sides = np.repeat([sizes], len(scales), axis=1).ravel()
    r_scales = np.repeat([scales], len(sizes), axis=0).ravel()
    ab_pyramid = np.transpose([r_sides * (r_scales ** .5) // 2,
                               r_sides / (r_scales ** .5) // 2]).astype(int)

    # Creates combinations of all anchor boxes centers and sides. Another product vectorization
    r_centers = np.repeat(centers, len(ab_pyramid), axis=0)
    r_ab_pyramid = np.repeat([ab_pyramid], len(centers), axis=0).reshape((-1, 2))
    return np.hstack(r_centers, r_ab_pyramid)


def valid_anchor_boxes(anchor_boxes, image_size):
    """Return indices of valid anchor boxes
    Receives:
        anchor_boxes [N, x, y, w, h] - all anchor boxes matrix
        image_size (W, H)
    Returns:
        indices [N]
        """
    img_width, img_height = image_size
    x, y, width, height = np.transpose(anchor_boxes)

    # Indicator matrix
    indicators = np.array([x - width // 2 >= 0,
                           y - height // 2 >= 0,
                           x + width // 2 <= img_width,
                           y + height // 2 <= img_height]).transpose()

    # Get indices of anchor boxes inside image
    return np.nonzero(np.all(indicators, axis=0, keepdims=False))


def compute_iou(anchor_boxes, gt_boxes):
    """Computes IoU for every anchor box and a batch of ground-truth boxes
    Receives:
        anchor_boxes [N, x, y, w, h]
        gt_boxes [M, x0, y0, x1, y1]
    Returns:
        ious [N] - max ious for avery anchor box among all gt boxes
        gt_boxes_index [N] - indices of chosen gt boxes"""
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
    gt_index = np.argmax(ious, axis=1)
    ious = np.take(ious, gt_index).ravel()
    return ious, gt_index


def compute_deltas(anchor_boxes, gt_boxes):
    """Computes special deltas between anchor boxes and respective gt boxes
    Receives:
        anchor_boxes [N, x, y, w, h]
        gt_boxes [N, x0, y0, x1, y1]
    Returns:
        deltas [N, (gt_x_center - x) / w,
                   (gt_y_centet - y) / h,
                   log(gt_width / w),
                   log(gt_height / h)] - RPN-required deltas"""
    x, y, width, height = np.transpose(anchor_boxes)
    x0, y0, x1, y1 = np.transpose(gt_boxes)

    # Deltas are computed for boxes in 'center' format
    gt_width = x1 - x0
    gt_height = y1 - y0
    gt_x_center = x0 + gt_width // 2
    gt_y_center = y0 + gt_height // 2
    return np.array([(gt_x_center - x) / width,
                     (gt_y_center - y) / height,
                     np.log(gt_width / width),
                     np.log(gt_height / height)])


def generate_labels_and_deltas(gt_boxes, anchor_boxes,
                               valid_indices,
                               lower_iou_threshold,
                               upper_iou_threshold,
                               max_positive_samples,
                               max_negative_samples,
                               random_generator):
    """Generate RPN training targets - labels and deltas
    Receives:
        gt_boxes [M, x0, y0, x1, y1]
        anchor_boxes [N, x, y, w, h]
        valid_indices [N] - indices of anchor boxes which will be used for computations
        lower_iou_threshold - Any anchor boxes with IoU above this threshold get '1' label (foreground)
        upper_iou_threshold - Any anchor boxes with IoU below this threshold get '1' label (foreground)
        max_positive_samples - maximum anchor boxes labeled as '1'
        max_negative_samples - maximum anchor boxes labeled as '0'
        random_generator - np random generator
    Returns:
        labels [N]
        deltas [N, dx, dy, dw, dh]"""
    ious = np.zeros(anchor_boxes[0])

    # Compute IoU for valid anchor boxes and get indices of respective gt boxes
    ious[valid_indices], gt_boxes_indices = compute_iou(anchor_boxes[valid_indices], gt_boxes)
    labels = np.full(ious.shape, -1)

    # Positive samples
    positive_indices = np.nonzero((ious > upper_iou_threshold)[valid_indices])
    if len(positive_indices) > max_positive_samples:
        positive_indices = random_generator.choice(positive_indices, max_positive_samples, replace=False)
    labels[positive_indices] = 1

    # Negative samples
    negative_indices = np.nonzero((ious < lower_iou_threshold)[valid_indices])
    if len(negative_indices) > max_negative_samples:
        negative_indices = random_generator.choice(negative_indices, max_negative_samples, replace=False)
    labels[negative_indices] = 0

    # Deltas are computed only for positive samples
    gt_boxes = np.take(gt_boxes,
                       gt_boxes_indices[ious[valid_indices] > upper_iou_threshold],
                       axis=0)
    deltas = np.zeros_like(anchor_boxes)
    deltas[positive_indices] = compute_deltas(anchor_boxes[positive_indices],
                                              gt_boxes)

    return labels, deltas


def slow_rpn_generator(pretrained_model,
                       image_data_generator,
                       *,
                       scales, sizes, seed,
                       lower_iou_threshold,
                       upper_iou_threshold,
                       max_positive_samples,
                       max_negative_samples,
                       ):
    """Generates RPN targets in a traditional way"""
    # Receive image size and output layer size from pretrained model
    image_size = pretrained_model.input_shape[1:3]
    feature_map_size = pretrained_model.output_shape[1:3]

    # Generate all anchor boxes beforehand
    all_ab = generate_anchor_boxes(sizes, scales,
                                   image_size,
                                   feature_map_size)
    valid_ab_indices = valid_anchor_boxes(all_ab,
                                          image_size)
    random_generator = np.random.Generator(seed)
    for img_batch, gt_boxes_batch in image_data_generator:
        # RPN receives feature maps
        feature_maps = pretrained_model.predict(img_batch)
        # TODO(Mocurin)
        labels, deltas = np.apply_along_axis(generate_labels_and_deltas,
                                             gt_boxes_batch, 0,
                                             all_ab, valid_ab_indices,
                                             lower_iou_threshold,
                                             upper_iou_threshold,
                                             max_positive_samples,
                                             max_negative_samples,
                                             random_generator)
        labels = np.squeeze(labels)
        deltas = np.squeeze(deltas)
        yield feature_maps, [labels], [deltas, labels]


def greedy_non_maximum_suppression(anchor_boxes, scores, overlap_threshold):
    """NMS, process RPN results to reduce useless and misleading regions for ROIP
    Receives:
        anchor_boxes [N, x, y, w, h] - delta-adjusted anchor boxes
        scores [N] - binary classification scores
        overlap_threshold - maximum regions overlap
    Возвращает:
        picked_anchor_boxes [M, x, y, w, h]
        picked_scores - [M]"""
    x, y, width, height = np.transpose(anchor_boxes)
    areas = width * height
    w_shift = width // 2
    h_shift = height // 2

    # Transform anchor boxes to 'corners' format
    x0 = x - w_shift
    y0 = y - h_shift
    x1 = x + w_shift
    y1 = y + h_shift

    sorted_indices = np.argsort(scores.ravel())
    picked_indices = []
    while len(sorted_indices) > 0:
        # Take anchor box with highest score
        last = sorted_indices[-1]
        sorted_indices = sorted_indices[:-1]
        picked_indices.append(last)

        # Find intersection of taken anchor box with others
        left = np.maximum(x0[last], x0[sorted_indices])
        top = np.maximum(y0[last], y0[sorted_indices])
        right = np.minimum(x1[last], x1[sorted_indices])
        bottom = np.minimum(y1[last], y1[sorted_indices])

        # Drop anchor boxes beyond overlap threshold
        overlap = np.maximum(0, right - left) * np.maximum(0, top - bottom) / areas[sorted_indices]
        sorted_indices = sorted_indices[overlap.ravel() < overlap_threshold]
    return anchor_boxes[picked_indices], scores[picked_indices]


def apply_deltas(all_anchor_boxes, all_deltas):
    """Layer which applies deltas to anchor boxes
    !!!Does not work with batches, needs to be applied along 0 axis!!!
    Receives:
        all_anchor_boxes [N, x, y, w. h]
        all_deltas [N, dx, dy, dw, dh] - respective deltas
    Returns:
        anchor_boxes [N, x, y, w, h]"""
    x, y, width, height = np.transpose(all_anchor_boxes)
    dx, dy, dw, dh = np.transpose(all_deltas)
    return np.transpose([x + dx * width,
                         y + dy * height,
                         width * tf.exp(dw),
                         height * tf.exp(dh)])