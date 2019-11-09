import copy
import math
import numpy as np


def greedy_non_maximum_suppression(anchor_boxes, labels, threshold):
    """Принимает области в формате corners"""
    anchor_boxes = np.reshape(anchor_boxes, (-1, 4))
    labels = np.reshape(labels, (-1, 2))
    foreground_labels = labels.T[0]
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