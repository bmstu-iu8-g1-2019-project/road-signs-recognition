import tensorflow as tf
from keras.layers import Layer


class RegionOfInterestPooling(Layer):
    """Region of interest pooling layer implementation
    Receives:
        input [feature_maps, regions]:
            feature_maps [batch_size, feature_map] - fms from feature extractor
            regions [batch_size, [regions_number, [x, y, w, h]]] - regions after NMSa
        parameters:
            target_size (target_width, target_height) - output fm shape
            image_size (image_width, image_height) - necessary as we are computing coordinates on
                                                     feature map, rather then on input image
            pooling_type 'max'/'min'/'mean' - pooling function to apply
    Returns:
        pooled_regions [batch_size, [regions_number, fm_crop]]"""
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
        feature_maps = input[0]
        regions = input[1]
        batch_size, fm_width, fm_height = tf.shape(feature_maps)[:3]

        def make_corners(regs):
            x, y, w, h = tf.unstack(regs, axis=1)
            w_str = w // 2
            h_str = h // 2
            return tf.stack([x - w_str,
                             y - h_str,
                             x + w_str,
                             y + h_str], axis=1)
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
    def __init__(self, overlap_threshold, regions_number, **kwargs):
        self.overlap_threshold = overlap_threshold
        self.regions_number = regions_number
        super(NonMaximumSuppression, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.regions_number, 4

    def call(self, input):
        def nms_wrapper(predictions):
            scores = predictions[:, 0]
            regions = predictions[:, 1:]
            indices = self._nms(scores, regions)
            return predictions[indices, 2:]
        # Does work with fixed regions number. Now does not
        return tf.map_fn(nms_wrapper, input)

    def _nms(self, scores, regions):
        left, top, right, bottom = tf.unstack(regions, axis=1)
        areas = (right - left) * (bottom - top)

        sorted_indices = tf.argsort(scores)
        picked_indices = []
        while tf.size(sorted_indices) > 0:
            # Take anchor box with highest score
            last = sorted_indices[-1]
            sorted_indices = sorted_indices[:-1]
            picked_indices.append(last)

            # Find intersection of taken anchor box with others
            left = tf.maximum(left[last], left[sorted_indices])
            top = tf.maximum(top[last], top[sorted_indices])
            right = tf.minimum(right[last], right[sorted_indices])
            bottom = tf.minimum(bottom[last], bottom[sorted_indices])

            # Drop anchor boxes beyond overlap threshold
            overlap = tf.maximum(0, right - left) * tf.maximum(0, top - bottom) / areas[sorted_indices]
            sorted_indices = sorted_indices[overlap.ravel() < self.overlap_threshold]
        return picked_indices


class ValidateRegions(Layer):
    def __init__(self, all_anchor_boxes, **kwargs):
        self.anchor_boxes = all_anchor_boxes
        super(ValidateRegions, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        # For sake of convenience merges scores and deltas tensor
        return input_shape[0], 5

    def call(self, input):
        scores = input[0]
        deltas = input[1]

        def apply_deltas(inp):
            x, y, w, h = tf.unstack(self.anchor_boxes, axis=1)
            dx, dy, dw, dh = tf.unstack(inp, axis=1)
            return tf.stack([x + dx * w,
                             y + dy * h,
                             w * tf.exp(dw),
                             h * tf.exp(dh)], axis=1)

        regions = tf.map_fn(apply_deltas, deltas)
        return tf.concat([scores[:, :, tf.newaxis], regions], axis=2)
