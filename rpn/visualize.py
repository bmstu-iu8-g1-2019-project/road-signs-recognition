from keras.preprocessing.image import load_img, img_to_array
from sources.feature_extractor.processing import prepare_feature_extractor
from sources.rpn.generation import RPNconfig
from keras.models import Model, load_model
from sources.rpn.rpn import clean_rpn_model, ThresholdedRegularizer, make_cls_wrapper, make_reg_wrapper, \
    expanded_sigmoid
from PIL import Image, ImageDraw, ImageFont
from sources.layers import ApplyDeltas, IndNonMaximumSuppression
import tensorflow as tf
import pandas as pd
import random as rd
import numpy as np


def center_to_corners(regions):
    y, x, h, w = np.transpose(regions)
    x0 = x - w // 2
    y0 = y - h // 2
    x1 = x + w // 2
    y1 = y + h // 2
    return np.transpose([y0, x0, y1, x1]).astype(int)


def prep_img(path, image_size):
    image = load_img(path, target_size=image_size, interpolation='lanczos')
    image.show()
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return np.divide(image, 255.)


if __name__ == '__main__':
    rpn_config = RPNconfig.load_json('versions/RPN_v26/rpn_config.json')
    cls_wrapper = make_cls_wrapper(tf.keras.losses.BinaryCrossentropy())
    reg_wrapper = make_reg_wrapper(tf.keras.losses.Huber())
    font = ImageFont.truetype('../misc/arial.ttf', size=8)
    rpn = load_model('versions/RPN_v26/configs/best.h5',
                     custom_objects={'ThresholdedRegularizer': ThresholdedRegularizer,
                                     'cls_wrapper': cls_wrapper,
                                     'reg_wrapper': reg_wrapper,
                                     'expanded_sigmoid': expanded_sigmoid})
    layer = ApplyDeltas(rpn_config.anchor_boxes,
                        rpn_config.valid_indices)([rpn.output[0],
                                                   rpn.output[1]])
    nms = IndNonMaximumSuppression(iou_threshold=0.5,
                                   rois_number=64)(layer)
    model = Model(inputs=rpn.input, outputs=[layer, nms, rpn.output[0]])
    rpn.summary()

    rd.seed(42)
    df = pd.read_json('../dataset/train.json')
    df = df.reset_index(drop=True)
    valid_ab = rpn_config.anchor_boxes[rpn_config.valid_indices]
    while True:
        index = rd.randint(0, df.shape[0] - 1)
        filename = df.loc[index]['filename']
        gt_boxes = df.loc[index]['gt_boxes']
        image = prep_img('../dataset/train/' + filename,  (720, 1280))

        regions, indices, cls = model.predict(image)
        indices = np.squeeze(indices)
        cls = np.squeeze(cls)[indices]

        center_ab = valid_ab[indices]
        corners_ab = center_to_corners(center_ab)

        center_reg = np.squeeze(regions)[indices, 1:].astype(int)
        corners_reg = center_to_corners(center_reg)

        image = Image.open('../dataset/train/' + filename)
        font = ImageFont.truetype('../misc/arial.ttf', size=8)
        draw = ImageDraw.Draw(image)
        for y0, x0, y1, x1 in corners_ab:
            draw.rectangle([(x0, y0), (x1, y1)], outline='orange')
        for y0, x0, y1, x1 in np.hstack([center_ab[:, :2], center_reg[:, :2]]):
            draw.line([(x0, y0), (x1, y1)], fill='yellow')
        for i, reg in enumerate(corners_reg):
            y0, x0, y1, x1 = reg
            draw.rectangle([(x0, y0), (x1, y1)], outline='red')
            draw.text((x0 + 3, y0 + 1), text=str(round(cls[i], 2)), font=font, fill='red')
        for y0, x0, y1, x1 in gt_boxes:
            draw.rectangle([(x0, y0), (x1, y1)], outline='green')
        image.show()
        input('Press F')





