from rpn.generation import compute_deltas
import numpy as np


def make_corners(regions):
    """Transforms boxes from center format to corners"""
    x0, y0, x1, y1 = np.transpose(regions)
    ws = x1 - x0 // 2
    hs = y1 - y0 // 2
    return np.transpose([x0 + ws, y0 + hs, 2 * ws, 2 * hs])


def compute_iou(regions, gt_boxes):
    """IOU metric computation. Differs from rpn version as there is no place for precomputation
    Also returns indices instead of respective gt boxes as classes are chosen with their help"""
    regions_areas = regions[:, 2] * regions[:, 3]
    x0, y0, x1, y1 = make_corners(regions)
    ious = []
    for gt_box in gt_boxes:
        gt_x0, gt_y0, gt_x1, gt_y1 = gt_box
        gt_area = (gt_x1 - gt_x0) * (gt_y1 - gt_y0)
        int_x0 = np.maximum(gt_x0, x0)
        int_y0 = np.maximum(gt_y0, y0)
        int_x1 = np.minimum(gt_x1, x1)
        int_y1 = np.minimum(gt_y1, y1)
        int_area = np.maximum(0, int_x1 - int_x0) * np.maximum(0, int_y1 - int_y0)
        ious.append(int_area / (regions_areas + gt_area - int_area))
    ious = np.transpose(ious)
    gt_indices = np.argmax(ious, axis=1)
    ious = np.squeeze(np.take_along_axis(ious, gt_indices[:, np.newaxis], axis=1))
    return ious, gt_indices


def create_generator(class_indices_dict, background_label='not_sign', threshold=0.4, ratio=1., seed=42):
    """Assigns labels for regions with gt_box IoU > threshold. Returns single targets array:
    class labels are connected horizontally with deltas"""
    background_index = class_indices_dict[background_label]
    random_generator = np.random.default_rng(seed)

    def pad_with_zeros(index):
        arr = np.zeros(len(class_indices_dict))
        arr[index] = 1
        return arr

    def targets_generator(regions, gt_boxes, gt_labels):
        ious, gt_indices = compute_iou(regions, gt_boxes)
        gt_boxes = np.take(gt_boxes, gt_indices, axis=0)

        bg_ind = np.flatnonzero(ious < threshold)
        fg_ind = np.flatnonzero(ious >= threshold)
        bg_samples = round(len(regions) / (1 + ratio))

        # Generator prioritizes fg samples by default
        if len(bg_ind) > bg_samples:
            bg_ind = random_generator.choice(bg_ind, bg_samples, replace=False)

        gt_labels = np.take(gt_labels, gt_indices, axis=0)
        # Convert labels to indices
        gt_labels = [class_indices_dict[label] for label in gt_labels]
        # Mark negative boxes as bg
        gt_labels[bg_ind] = background_index
        gt_labels = np.apply_along_axis(pad_with_zeros, 0, gt_labels[:, np.newaxis])

        deltas = np.zeros_like(regions, dtype='float')
        deltas[fg_ind] = compute_deltas(regions[fg_ind], gt_boxes[fg_ind])
        return np.hstack([gt_labels, deltas])
    return targets_generator


def detector_generator(image_generator, rpn_model, class_indices_dict, *args, **kwargs):
    # Keras does not allow to generate labels on output from intermediate layer, so generator should receive
    # whole RPN model, which makes fine-tuning pretrained model with this generator difficult (if not impossible)
    roip_crops, fm_crop_width, fm_crop_height, fm_crop_depth = rpn_model.outputs[0].shape[1:]
    targets_generator = create_generator(class_indices_dict, *args, **kwargs)
    for img_batch, gt_boxes_batch, gt_labels_batch in image_generator:
        fm_crops, regions_batch = rpn_model.predict_on_batch(img_batch)
        # Unfold batch dimension. Detector is applied to every region, rather than a batch of regions
        fm_crops = np.reshape(fm_crops, (-1, fm_crop_width, fm_crop_height, fm_crop_depth))
        targets = np.array([targets_generator(regions, gt_boxes, gt_labels)
                            for regions, gt_boxes, gt_labels
                            in zip(regions_batch, gt_boxes_batch, gt_labels_batch)])
        # Unfolding once again
        targets = np.reshape(targets, (-1, len(class_indices_dict) + 4))
        # Actual (len(class_indices) * roip_crops)-sized matrix full of integers. You do not want your nms to output
        # 2k regions with this generator. But i still believe that making these arrays here is better than
        # in tensorflow graph
        labels = targets[:, len(class_indices_dict)].astype(int)
        yield fm_crops, [labels, targets]
