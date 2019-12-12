import tensorflow as tf
import keras.backend as K
from keras.layers import Layer


@tf.function
def make_corners(regs):
    x, y, w, h = tf.unstack(regs, axis=1)
    w_str = w // 2
    h_str = h // 2
    return tf.stack([x - w_str, y - h_str, x + w_str, y + h_str], axis=1)


class RegionOfInterestPooling(Layer):
    """Region of interest pooling layer implementation
    Parameters:
        target_size (target_width, target_height) - output feature map shape
        image_size (image_width, image_height) - necessary as we are computing coordinates on
                                                 feature map, rather then on input image
        pooling_type 'max'/'min'/'mean' - pooling function to apply
    Receives:
        feature_maps: (batch_size, fm_width, fm_height, fm_depth) - feature extractor output
        regions: (batch_size, rois_number, 4) - regions after NMS application
    Returns:
        output: (batch_size, rois_number, target_width, target_height, fm_depth)"""
    _pool_types = {'max': tf.math.reduce_max, 'min': tf.math.reduce_min, 'mean': tf.math.reduce_mean}

    def __init__(self, target_size, image_size, pooling_type='max', **kwargs):
        self.target_width, self.target_height = target_size
        self.image_width, self.image_height = image_size
        # Choose pooling function to apply
        assert pooling_type in self._pool_types.keys(), 'No such pooling type'
        self.pooling = self._pool_types[pooling_type]
        super(RegionOfInterestPooling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # ROIP receives batches of feature maps and regions
        feature_map_shape, regions_shape = input_shape
        batch_size = feature_map_shape[0]
        regions_number = regions_shape[1]
        depth = feature_map_shape[3]
        # ROIP outputs batches of pooled feature map crops
        return batch_size, regions_number, self.target_width, self.target_height, depth

    def call(self, input):
        feature_maps, regions = input
        batch_size, fm_width, fm_height = tf.shape(feature_maps)[:3]

        # Whole regions batch is now in 'corners' format
        regions = tf.map_fn(make_corners, regions)

        def make_fm_positions(regs):
            l, t, r, b = tf.unstack(regs, axis=1)
            return tf.stack([l / self.image_width * fm_width,
                             t / self.image_height * fm_height,
                             r / self.image_width * fm_width,
                             b / self.image_height * fm_height], axis=1)
        # Whole regions batch is now represented by respective fm coordinates
        regions = tf.map_fn(make_fm_positions, regions)

        def crop_fm_regions(fm, regs):
            regs = tf.unstack(regs)
            return tf.stack([fm[r[0]:r[2], r[1]:r[3], :] for r in regs])
        # Batch of feature map crops from every region
        cropped_fms = tf.stack([crop_fm_regions(feature_maps[i], regions[i])
                                for i in range(batch_size)])

        # Applies pooling to every list of crops in a batch
        return tf.map_fn(lambda crops: tf.map_fn(self._pool_crop, crops), cropped_fms)

    def _pool_crop(self, crop):
        w, h = tf.shape(crop)[:2]
        w_step = tf.cast(w / self.target_width, 'int32')
        h_step = tf.cast(h / self.target_height, 'int32')
        # Arranges a matrix with [x, y] containing respective pooling cell coordinates
        # Get rid of python loops?
        crop_grid = [[(i * h_step,
                       j * w_step,
                       (i + 1) * h_step if i + 1 < self.target_height else h,
                       (j + 1) * w_step if j + 1 < self.target_width else w)
                      for j in range(self.target_width)]
                     for i in range(self.target_height)]

        def pool_cell(reg):
            return self.pooling(crop[reg[0]:reg[2], reg[1]:reg[3], :], axis=[0, 1])

        # Convert to tf tensor
        return tf.stack([[pool_cell(cell_reg)
                          for cell_reg in grid_row]
                         for grid_row in crop_grid])


class NonMaximumSuppression(Layer):
    """Calls NMS for every sample and pads received region batches with random regions
    Parameters:
        iou_threshold, score_threshold, rois_number(max_output_size) are tf.non_maximum_suppression parameters
    Receives:
        input: (batch_size, valid_ab_number, 5) - tensor of scores and respective regions
    Returns:
        output: (batch_size, rois_number, 4) - NMS`ed regions tensor"""
    def __init__(self,
                 iou_threshold=0.5,
                 rois_number=256,
                 score_threshold=float('-inf'),
                 return_indices=False,
                 **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.rois_number = rois_number
        self.return_indices = tf.cast(return_indices, tf.bool)
        super(NonMaximumSuppression, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """NMS is padded, so layer always returns fixed number of regions"""
        return K.switch(self.return_indices, (input_shape[0], self.rois_number, 1), (input_shape[0], self.rois_number, 4))

    def call(self, input):
        def nms_wrapper(predictions):
            scores = predictions[:, 0]
            regions = make_corners(predictions[:, 1:])
            indices = tf.image.non_max_suppression(regions, scores,
                                                   self.rois_number,
                                                   self.iou_threshold,
                                                   self.score_threshold)
            def pad(ind):
                missing = self.rois_number - tf.size(ind)
                padding = tf.random.uniform([missing],
                                            minval=0,
                                            maxval=tf.shape(regions)[0],
                                            dtype=tf.int32)
                return tf.concat([ind, padding], axis=0)
            # Pad NMS`ed indices with random indices to make layer output shape fixed.
            result_indices = K.switch(tf.size(indices) < self.rois_number, pad(indices), indices)
            return K.switch(self.return_indices, result_indices, tf.gather(predictions, result_indices)[:, 1:])
        return tf.map_fn(nms_wrapper, input)


class IndNonMaximumSuppression(Layer):
    """Calls NMS for every sample and pads received region batches with random regions
    Parameters:
        iou_threshold, score_threshold, rois_number(max_output_size) are tf.non_maximum_suppression parameters
    Receives:
        input: (batch_size, valid_ab_number, 5) - tensor of scores and respective regions
    Returns:
        output: (batch_size, rois_number, 4) - NMS`ed regions tensor"""
    def __init__(self,
                 iou_threshold=0.5,
                 rois_number=256,
                 score_threshold=float('-inf'),
                 return_indices=False,
                 **kwargs):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.rois_number = rois_number
        super(IndNonMaximumSuppression, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """NMS is padded, so layer always returns fixed number of regions"""
        return input_shape[0], self.rois_number, 1

    def call(self, input):
        def nms_wrapper(predictions):
            scores = predictions[:, 0]
            regions = make_corners(predictions[:, 1:])
            indices = tf.image.non_max_suppression(regions, scores,
                                                   self.rois_number,
                                                   self.iou_threshold,
                                                   self.score_threshold)
            def pad(ind):
                missing = self.rois_number - tf.size(ind)
                padding = tf.random.uniform([missing],
                                            minval=0,
                                            maxval=tf.shape(regions)[0],
                                            dtype=tf.int32)
                return tf.concat([ind, padding], axis=0)
            # Pad NMS`ed indices with random indices to make layer output shape fixed.
            return K.switch(tf.size(indices) < self.rois_number, pad(indices), indices)
        return tf.map_fn(nms_wrapper, input, dtype=tf.int32)


class ApplyDeltas(Layer):
    """Picks valid regions, applies deltas for them, transforms regions to 'corners' format and merges
    scores and regions into one tensor
    Parameters:
        anchor_boxes: (ab_number, 4) - all image anchor boxes
        valid_anchor_boxes: (valid_ab_number) - indices of valid anchor boxes
    Receives:
        scores: (batch_size, ab_number) - RPN region scores
        deltas: (batch_size, ab_number, 4) - RPN region deltas
    Returns:
        regions: (batch_size, valid_ab_number, 4) - regions in [x, y, w, h] format"""
    def __init__(self, anchor_boxes, valid_indices, **kwargs):
        self.valid_anchor_boxes = tf.cast(anchor_boxes[valid_indices], tf.float32)
        self.valid_indices = valid_indices
        super(ApplyDeltas, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # For sake of convenience merges scores and regions tensor
        return input_shape[0], len(self.valid_indices), 5

    def call(self, input):
        scores, deltas = input
        scores = tf.gather(scores, self.valid_indices, axis=1)
        deltas = tf.gather(deltas, self.valid_indices, axis=1)
        x, y, w, h = tf.unstack(self.valid_anchor_boxes, axis=1)

        def apply_deltas(inp):
            dx, dy, dw, dh = tf.unstack(inp, axis=1)
            return tf.stack([x + dx * w, y + dy * h, w * tf.exp(dw), h * tf.exp(dh)], axis=1)
        regions = tf.map_fn(apply_deltas, deltas)
        return tf.concat([scores[:, :, tf.newaxis], regions], axis=2)

