from math import floor, ceil
import numpy as np
import json
import os


class RPNconfig:
    """Simple config holder with ability to save and load"""
    def __init__(self, image_size, fm_size, scales, sizes):
        self.image_size = image_size
        self.fm_size = fm_size
        self.sizes = sizes
        self.scales = scales
        self.anchors_per_fm_point = len(sizes) * len(scales)
        self.anchor_boxes, self.valid_indices = rpn_anchor_boxes(image_size,
                                                                 fm_size,
                                                                 sizes,
                                                                 scales)

    def save_json(self, path):
        assert os.path.splitext(path)[-1] == '.json', 'Config can only be a json'
        data = {'image_size': self.image_size,
                'fm_size': self.fm_size,
                'sizes': self.sizes,
                'scales': self.scales}
        with open(path, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def load_json(path):
        assert os.path.splitext(path)[-1] == '.json', 'Config can only be a json'
        with open(path, 'r') as file:
            data = json.load(file)
            return RPNconfig(**data)


def rpn_anchor_boxes(image_size, *args, **kwargs):
    """Generate anchor boxes and indices of valid anchor boxes beforehand"""
    anchor_boxes = generate_anchor_boxes(image_size, *args, **kwargs)
    valid_ab_indices = valid_anchor_boxes(anchor_boxes, image_size)
    return anchor_boxes, valid_ab_indices


def generate_anchor_boxes(image_size, feature_map_size, sizes, scales):
    """Generates all anchor boxes for current RPN configuration.
    Receives:
        sizes: sizes of 1:1 anchor boxes
        scales: sides ratios of anchor boxes
        image_size: (iH, iW)
        feature_map_size: (fH, fW)
    Returns:
        anchor_boxes: shape (N, 4) """
    image_height, image_width = image_size
    fm_height, fm_width = feature_map_size
    height_stride = int(image_height / fm_height)
    width_stride = int(image_width / fm_width)

    # Compose horizontal and vertical positions into grid and reshape result into (-1, 2)
    y_centers = np.arange(0, image_height, height_stride)
    x_centers = np.arange(0, image_width, width_stride)
    centers = np.dstack(np.meshgrid(y_centers, x_centers)).reshape((-1, 2))

    # Creates anchor boxes pyramid. Somewhat vectorized version of itertools.product
    r_scales = np.repeat([scales], len(sizes), axis=0).ravel()
    r_sides = np.repeat([sizes], len(scales), axis=1).ravel()
    ab_pyramid = np.transpose([r_sides / (r_scales ** .5),
                               r_sides * (r_scales ** .5)]).astype(int)

    # Creates combinations of all anchor boxes centers and sides
    r_centers = np.repeat(centers, len(ab_pyramid), axis=0)
    r_ab_pyramid = np.repeat([ab_pyramid], len(centers), axis=0).reshape((-1, 2))
    return np.hstack((r_centers, r_ab_pyramid))


def valid_anchor_boxes(anchor_boxes, image_size):
    """Return indices of valid anchor boxes,
    Anchor box is considered valid if it is inside image entirely
    Receives:
        anchor_boxes: shape (N, 4)
        image_size: (iH, iW)
    Returns:
        indices shape (M)
        """
    img_height, img_width = image_size
    y, x, height, width = np.transpose(anchor_boxes)

    # TODO(Mocurin) Optimize?
    # Indicator matrix
    indicators = np.array([y - height // 2 >= 0,
                           x - width // 2 >= 0,
                           y + height // 2 <= img_height,
                           x + width // 2 <= img_width]).transpose()

    # Get indices of anchor boxes inside image
    return np.flatnonzero(np.all(indicators, axis=1, keepdims=False))


def compute_deltas(anchor_boxes, gt_boxes):
    """Computes deltas between anchor boxes and respective gt boxes
    Receives:
        anchor_boxes: shape (N, 4) 'center' format
        gt_boxes: shape (N, 4) 'corners' format
    Returns:
        deltas: shape (n, 4)
        These 4 are:
            dy = gt_y_center - y) / h
            dx = gt_x_center - x) / w
            dh = log(gt_height / h)
            dw = log(gt_width / w)"""
    y, x, height, width = np.transpose(anchor_boxes)
    y0, x0, y1, x1 = np.transpose(gt_boxes)

    # Gt boxes should be in 'center' format
    gt_height = y1 - y0
    gt_width = x1 - x0
    gt_y_center = y0 + gt_height // 2
    gt_x_center = x0 + gt_width // 2
    return np.transpose([(gt_y_center - y) / height,
                         (gt_x_center - x) / width,
                         np.log(gt_height / height),
                         np.log(gt_width / width)])

# TODO(Mocurin) Unify metrics, same with detector


def create_pos_overlap_metric(anchor_boxes):
    """In RPN generation metric computation is happening for all valid anchor boxes,
    so it is done beforehand in this decorator
    PositiveOverlapMetric is a part of IoU metric.
    It is computed as a ratio of intersection area to positive region area (ground-truth box)"""
    y, x, h, w = np.transpose(anchor_boxes)
    y0 = y - h // 2
    x0 = x - w // 2
    y1 = y + h // 2
    x1 = x + w // 2

    def pos_overlap(gt_boxes):
        pos_overlaps = []
        for gt_box in gt_boxes:
            gt_y0, gt_x0, gt_y1, gt_x1 = gt_box
            gt_area = (gt_x1 - gt_x0) * (gt_y1 - gt_y0)
            int_y0 = np.maximum(gt_y0, y0)
            int_x0 = np.maximum(gt_x0, x0)
            int_y1 = np.minimum(gt_y1, y1)
            int_x1 = np.minimum(gt_x1, x1)
            int_area = np.maximum(0, int_x1 - int_x0) * np.maximum(0, int_y1 - int_y0)
            pos_overlaps.append(int_area / gt_area)
        # Group by anchor boxes
        pos_overlaps = np.transpose(pos_overlaps)
        # Get max metric index
        gt_indices = np.argmax(pos_overlaps, axis=1)
        # Choose max metric
        pos_overlaps = np.squeeze(np.take_along_axis(pos_overlaps, gt_indices[:, np.newaxis], axis=1))
        # Take respective ground-truth boxes. No reason to return indices, at least in RPN
        gt_boxes = np.take(gt_boxes, gt_indices, axis=0)
        return pos_overlaps, gt_boxes
    return pos_overlap


def create_overlap_metric(anchor_boxes):
    """In RPN generation metric computation is happening for all valid anchor boxes,
    so it is done beforehand in this decorator
    OverlapMetric is a part of IoU metric.
    It is computed as a ratio of intersection area to negative region area (anchor box)"""
    y, x, h, w = np.transpose(anchor_boxes)
    ab_area = w * h
    y0 = y - h // 2
    x0 = x - w // 2
    y1 = y + h // 2
    x1 = x + w // 2

    def overlap(gt_boxes):
        overlaps = []
        for gt_box in gt_boxes:
            gt_y0, gt_x0, gt_y1, gt_x1 = gt_box
            int_y0 = np.maximum(gt_y0, y0)
            int_x0 = np.maximum(gt_x0, x0)
            int_y1 = np.minimum(gt_y1, y1)
            int_x1 = np.minimum(gt_x1, x1)
            int_area = np.maximum(0, int_x1 - int_x0) * np.maximum(0, int_y1 - int_y0)
            overlaps.append(int_area / ab_area)
        overlaps = np.transpose(overlaps)
        gt_indices = np.argmax(overlaps, axis=1)
        overlaps = np.squeeze(np.take_along_axis(overlaps, gt_indices[:, np.newaxis], axis=1))
        gt_boxes = np.take(gt_boxes, gt_indices, axis=0)
        return overlaps, gt_boxes
    return overlap


def create_iou_metric(anchor_boxes):
    """In RPN generation metric computation is happening for all valid anchor boxes,
    so it is done beforehand in this decorator
    IoU is computed as a ratio of intersection area to a regions union area"""
    y, x, h, w = np.transpose(anchor_boxes)
    ab_areas = w * h
    y0 = y - h // 2
    x0 = x - w // 2
    y1 = y + h // 2
    x1 = x + w // 2

    def iou(gt_boxes):
        ious = []
        for gt_box in gt_boxes:
            gt_y0, gt_x0, gt_y1, gt_x1 = gt_box
            gt_area = (gt_x1 - gt_x0) * (gt_y1 - gt_y0)
            int_y0 = np.maximum(gt_y0, y0)
            int_x0 = np.maximum(gt_x0, x0)
            int_y1 = np.minimum(gt_y1, y1)
            int_x1 = np.minimum(gt_x1, x1)
            int_area = np.maximum(0, int_x1 - int_x0) * np.maximum(0, int_y1 - int_y0)
            ious.append(int_area / (ab_areas + gt_area - int_area))
        ious = np.transpose(ious)
        gt_indices = np.argmax(ious, axis=1)
        ious = np.squeeze(np.take_along_axis(ious, gt_indices[:, np.newaxis], axis=1))
        gt_boxes = np.take(gt_boxes, gt_indices, axis=0)
        return ious, gt_boxes
    return iou


_metrics = {'iou': create_iou_metric, 'positive_overlap': create_pos_overlap_metric, 'overlap': create_overlap_metric}


def create_fixed_generator(anchor_boxes, valid_indices,
                           lower_threshold, upper_threshold,
                           ratio=1., metric='iou', minibatch_size=256, seed=42):
    """Creates a generator with fixed batch size. Generator is choosing anchor boxes by thresholds, but if there
    is not enough samples, it just takes N max or N min. During choice process there is no full sort, just a partial
    sort, so it will be faster. Sorting is NOT happening if after thresholds application there are enough examples.
    If there is more samples, than required, targets are chosen randomly"""
    assert minibatch_size <= len(valid_indices), 'Minibatch length must be greater than valid regions number'
    assert metric in _metrics.keys(), 'Only available metrics are \'iou\', \'positive_overlap\' and \'overlap\''
    valid_ab = anchor_boxes[valid_indices]
    compute_metric = _metrics[metric](valid_ab)
    neg_samples = floor(minibatch_size / (1 + ratio))
    pos_samples = ceil(neg_samples * ratio)
    targets_shape = (len(anchor_boxes), 5)
    random_generator = np.random.default_rng(seed=seed)

    def targets_generator(gt_boxes):
        metrics, gt_boxes = compute_metric(gt_boxes)
        neg_ind = np.flatnonzero(metrics < lower_threshold)
        pos_ind = np.flatnonzero(metrics > upper_threshold)

        if len(neg_ind) > neg_samples:
            neg_ind = random_generator.choice(neg_ind, neg_samples, replace=False)
        elif len(neg_ind) < neg_samples:
            neg_ind = np.argpartition(metrics, neg_samples)[:neg_samples]
        if len(pos_ind) > pos_samples:
            pos_ind = random_generator.choice(pos_ind, pos_samples, replace=False)
        elif len(pos_ind) < pos_samples:
            pos_ind = np.argpartition(metrics, len(metrics) - pos_samples)[-pos_samples:]
        labels = np.full_like(metrics, -1, dtype='int')
        labels[pos_ind] = 1
        labels[neg_ind] = 0

        deltas = np.full_like(gt_boxes, 0, dtype='float')
        deltas[pos_ind] = compute_deltas(valid_ab[pos_ind], gt_boxes[pos_ind])

        targets = np.zeros(targets_shape, dtype='float')
        targets[:, 0] = -1
        targets[valid_indices] = np.hstack([labels[:, np.newaxis], deltas])
        # Since there is no way to give a loss function two tensors,
        # we have to make one, containing all required labels
        return targets
    return targets_generator


def create_varying_generator(anchor_boxes, valid_indices,
                             lower_threshold, upper_threshold,
                             ratio=1., metric='iou', seed=42):
    """Creates varying length batches. Metric is computed, thresholds are applied.
    If there is more samples, than required, targets are chosen randomly"""
    assert metric in _metrics.keys(), 'Only available metrics are \'iou\', \'positive_overlap\' and \'overlap\''
    valid_ab = anchor_boxes[valid_indices]
    compute_metric = _metrics[metric](valid_ab)
    targets_shape = (len(anchor_boxes), 5)
    random_generator = np.random.default_rng(seed=seed)

    def targets_generator(gt_boxes):
        metrics, gt_boxes = compute_metric(gt_boxes)
        neg_ind = np.flatnonzero(metrics < lower_threshold)
        pos_ind = np.flatnonzero(metrics > upper_threshold)

        if len(pos_ind) < len(neg_ind):
            neg_samples = round(len(pos_ind) / ratio)
            neg_ind = random_generator.choice(neg_ind, neg_samples, replace=False)
        elif len(neg_ind) < len(pos_ind):
            pos_samples = round(len(neg_ind) * ratio)
            pos_ind = random_generator.choice(pos_ind, pos_samples, replace=False)
        labels = np.full_like(metrics, -1, dtype='int')
        labels[pos_ind] = 1
        labels[neg_ind] = 0

        deltas = np.full_like(gt_boxes, 0, dtype='float')
        deltas[pos_ind] = compute_deltas(valid_ab[pos_ind], gt_boxes[pos_ind])

        targets = np.zeros(targets_shape, dtype='float')
        targets[:, 0] = -1.
        targets[valid_indices] = np.hstack([labels[:, np.newaxis], deltas])
        return targets
    return targets_generator


_gen_types = {'varying': create_varying_generator, 'fixed': create_fixed_generator}


def rpn_generator(data_generator, gen_type, *args, **kwargs):
    assert gen_type in _gen_types.keys(), 'Only available types are \'fixed\' and \'varying\''
    targets_generator = _gen_types[gen_type](*args, **kwargs)
    for imgs_batch, gt_boxes_batch in data_generator:
        targets = np.array([targets_generator(gt_box) for gt_box in gt_boxes_batch])
        labels = targets[:, :, 0].astype(int)
        yield imgs_batch, [labels, targets]
