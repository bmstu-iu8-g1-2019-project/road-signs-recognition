import copy
import math
import numpy as np


def generate_point_anchor_boxes(*, position, scales, aspect_ratios):
    return [
        [*position,
         math.floor(sc * math.sqrt(ar)),
         math.floor(sc / math.sqrt(ar))]
        for sc in scales
        for ar in aspect_ratios
    ]


def generate_anchors_grid(image_size, feature_map_size):
    h_subratio, v_subratio = np.ceil(
        np.divide(image_size, feature_map_size)
    ).astype(int)
    width, height = image_size
    x = np.arange(1, width, h_subratio)
    y = np.arange(1, height, v_subratio)
    return np.meshgrid(x, y)


def generate_feature_map_anchor_boxes(image_size, feature_map_size, *,
                                      scales, aspect_ratios):
    paired = np.dstack(generate_anchors_grid(image_size, feature_map_size))
    results = []
    for row in paired:
        for pair in row:
            results.append(
                generate_point_anchor_boxes(
                    position=pair,
                    scales=scales,
                    aspect_ratios=aspect_ratios)
            )
    return np.reshape(
        results,
        (*feature_map_size,
         len(scales) * len(aspect_ratios),
         4)
    )


def region_area(region):
    x0, y0, x1, y1 = region
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def region_intersection(first_region, second_region):
    x0, y0, x1, y1 = first_region
    x2, y2, x3, y3 = second_region
    return [max(x0, x2),
            max(y0, y2),
            min(x1, x3),
            min(y1, y3)]


# Anchor box must be in two points format
def compute_iou(first_region, second_region):
    intersection = region_intersection(first_region, second_region)
    intersection_area = region_area(intersection)
    first_region_area = region_area(first_region)
    second_region_area = region_area(second_region)
    return intersection_area / float(
        first_region_area + second_region_area - intersection_area
    )


# Generate list of max iou for every anchor box in shape (1, -1)
def compute_all_iou(anchor_boxes, ground_truth_boxes):
    all_iou = [[compute_iou(a_box, gt_box)
                for a_box in anchor_boxes]
               for gt_box in ground_truth_boxes]
    return np.amax(all_iou, axis=0)


# Generate list of anchor boxes indices in shape (-1, 3)
def get_anchor_boxes_indices(anchors_tensor):
    indices = np.indices(anchors_tensor.shape[:3])
    return np.hstack((indices[0].reshape(-1, 1),
                      indices[1].reshape(-1, 1),
                      indices[2].reshape(-1, 1)))


# There are 3 ab formats:
# Center - one we are using in RPN. (x_center, y_center, width, height)
# UpperLeft - one we have in dataset (x_upper_left, y_upper_left, width, height)
# Corners - one we are using for ROI computation
# TODO(Mocurin): Get rid of this abomination
def reformat(anchor_box, *, from_type, to_type):
    assert len(anchor_box) == 4, 'Anchor is 4-sized array'
    box = copy.deepcopy(anchor_box)
    if from_type == 'center':
        box[0] = round(box[0] - box[2] / 2)
        box[1] = round(box[1] - box[3] / 2)
        if to_type == 'upper_left':
            return box
        elif to_type == 'corners':
            box[2] += box[0]
            box[3] += box[1]
            return box
    elif from_type == 'corners':
        box[2] -= box[0]
        box[3] -= box[1]
        if to_type == 'upper_left':
            return box
        elif to_type == 'center':
            box[0] = round(box[0] + box[2] / 2)
            box[1] = round(box[1] + box[3] / 2)
            return box
    elif from_type == 'upper_left':
        if to_type == 'corners':
            box[2] += box[0]
            box[3] += box[1]
            return box
        elif to_type == 'center':
            box[0] = round(box[0] + box[2] / 2)
            box[1] = round(box[1] + box[3] / 2)
            return box


def validate_format(boxes, *, from_type, to_type):
    formatted_boxes = np.reshape(boxes, (-1, 4))
    if from_type != to_type:
        formatted_boxes = np.apply_along_axis(
            reformat, axis=1, arr=formatted_boxes,
            from_type=from_type,
            to_type=to_type)
    return formatted_boxes


# only for np.argwhere
def are_valid_anchor_boxes(anchor_boxes, image_size):
    width, height = image_size
    return [x_left > 0 and
            y_top > 0 and
            x_right < width and
            y_bottom < height
            for x_left, y_top, x_right, y_bottom in anchor_boxes]


def validate_anchors(anchor_boxes, anchor_indices, *, image_size):
    valid_indices = np.argwhere(are_valid_anchor_boxes(anchor_boxes, image_size)).ravel()
    return anchor_boxes.take(valid_indices, axis=0), anchor_indices.take(valid_indices, axis=0)


#
def shuffle_anchor_boxes(anchor_boxes, indices):
    shuffled_indices = range(len(anchor_boxes))
    np.random.shuffle(shuffled_indices)
    return [anchor_boxes.take(shuffled_indices, axis=0),
            indices.take(shuffled_indices, axis=0)]


# Generate boxes batch from anchor boxes and ground truth boxes
# Generate indices in shape (3, -1)
# Generate anchor boxes in shape (4, -1)
# Generate labels in shape (1, -1)
def generate_batch(anchor_boxes, ground_truth_boxes, *,
                   boxes_format, image_size,
                   lower_threshold=0.3, upper_threshold=0.7,
                   batch_size=256, track_ious=False):
    indices = get_anchor_boxes_indices(anchor_boxes)
    a_boxes = validate_format(anchor_boxes,
                              from_type=boxes_format[0],
                              to_type='corners')
    gt_boxes = validate_format(ground_truth_boxes,
                               from_type=boxes_format[1],
                               to_type='corners')
    a_boxes, indices = validate_anchors(a_boxes, indices, image_size=image_size)
    a_boxes, indices = shuffle_anchor_boxes(a_boxes, indices)
    ious = compute_all_iou(a_boxes, gt_boxes)
    sort_indices = np.argsort(ious, axis=0)
    sort_indices = np.hstack((sort_indices[:batch_size // 2],
                             sort_indices[-batch_size // 2:]))
    ious = ious.take(sort_indices)
    labels = np.full(batch_size, -1)
    labels[ious > upper_threshold] = 1
    labels[ious < lower_threshold] = 0
    labels = labels.reshape(-1, 1)
    if track_ious:
        ious = ious.reshape(-1, 1)
        labels = np.hstack((labels, ious))
    return a_boxes.take(sort_indices, axis=0), indices.take(sort_indices, axis=0), labels


def greedy_non_maximum_suppression(anchor_boxes, labels, threshold):
    anchor_boxes = np.reshape(anchor_boxes, (-1, 4))
    labels = np.reshape(labels, (-1, 2))
    foreground_labels = labels.T[0]
    anchor_boxes = np.apply_along_axis(reformat, axis=1,
                                       arr=anchor_boxes,
                                       from_type='upper_left',
                                       to_type='corners')
    x0 = anchor_boxes.T[0]
    y0 = anchor_boxes.T[1]
    x1 = anchor_boxes.T[2]
    y1 = anchor_boxes.T[3]
    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    sorted_indices = np.argsort(foreground_labels)
    picked_indices = []
    while len(sorted_indices) > 0:
        last = sorted_indices[-1]
        sorted_indices = sorted_indices[:-1]
        picked_indices.append(last)
        x0_tmp = np.maximum(x0[last], x0[sorted_indices])
        y0_tmp = np.maximum(y0[last], y0[sorted_indices])
        x1_tmp = np.minimum(x1[last], x1[sorted_indices])
        y1_tmp = np.minimum(y1[last], y1[sorted_indices])
        width = np.maximum(0, x1_tmp - x0_tmp + 1)
        height = np.maximum(0, y1_tmp - y0_tmp + 1)
        overlap = (width * height) / areas[sorted_indices]
        sorted_indices = sorted_indices.take(np.argwhere(overlap < threshold).ravel(), axis=0)
    return anchor_boxes[picked_indices], labels[picked_indices]
