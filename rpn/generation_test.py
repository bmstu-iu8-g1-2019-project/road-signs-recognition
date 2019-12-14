from PIL import Image, ImageDraw, ImageFont
from rpn.generation import rpn_anchor_boxes, create_fixed_generator
import pandas as pd
import numpy as np
import random as rd


_font = ImageFont.truetype('../misc/arial.ttf', size=16)


def visalize_anchor_boxes(image_size, feature_map_size, size, scale, valid=False):
    image = Image.new('RGB', (image_size[1], image_size[0]))
    draw = ImageDraw.Draw(image)
    anchor_boxes, valid_indices = rpn_anchor_boxes(image_size, feature_map_size, [size], [scale])
    if valid:
        anchor_boxes = anchor_boxes[valid_indices]
    for ab in anchor_boxes:
        y, x, h, w = ab
        draw.rectangle([x - w // 2, y - h // 2, x + w // 2, y + h // 2], outline='red')
    image.show()


seed = 42


def visualize_labels_gen(path,
                         dataframe,
                         anchor_boxes,
                         valid_indices,
                         *args, **kwargs):
    generator = create_fixed_generator(anchor_boxes, valid_indices, *args, **kwargs)
    rd.seed(seed)
    while True:
        index = rd.randint(0, dataframe.shape[0] - 1)
        filename = dataframe.loc[index]['filename']
        gt_boxes = dataframe.loc[index]['gt_boxes']
        targets = generator(gt_boxes)
        pos_ind = np.where(np.equal(targets[:, 0], 1))
        neg_ind = np.where(np.equal(targets[:, 0], 0))
        pos_deltas = np.take(targets[:, 1:], pos_ind, axis=0)
        pos_ab = np.take(anchor_boxes, pos_ind, axis=0)
        neg_ab = np.take(anchor_boxes, neg_ind, axis=0)

        def apply_and_transform(deltas, ab):
            dy, dx, dh, dw = np.transpose(deltas)
            y, x, h, w = np.transpose(ab)
            x1 = x + dx * w
            y1 = y + dy * h
            ws = w * np.exp(dw) // 2
            hs = h * np.exp(dh) // 2
            return np.transpose([y1 - hs, x1 - ws, y1 + hs, x1 + ws])
        pos_boxes = apply_and_transform(pos_deltas, pos_ab).reshape((-1, 4))

        def transform(ab):
            y, x, h, w = np.transpose(ab)
            ws = w // 2
            hs = h // 2
            return np.transpose([y - hs, x - ws, y + hs, x + ws])
        neg_boxes = transform(neg_ab).reshape(-1, 4)
        image = Image.open(path + filename)
        draw = ImageDraw.Draw(image)
        for box in gt_boxes:
            y0, x0, y1, x1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline='blue')
        for box in pos_boxes:
            y0, x0, y1, x1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline='green')
        for box in neg_boxes:
            y0, x0, y1, x1 = box
            draw.rectangle([(x0, y0), (x1, y1)], outline='red')
        image.show()
        input('Got!')


if __name__ == '__main__':
    visalize_anchor_boxes((720, 1280), (45, 80), 24, 1.)
    ab, valid = rpn_anchor_boxes((720, 1280), (45, 80), [24, 48, 96], [0.5, 1., 1.5])
    df = pd.read_json('../dataset/test.json')
    df = df.reset_index(drop=True)
    print(df.head())
    visualize_labels_gen('../dataset/test/',
                         df, ab, valid, 0.05, 0.4, 1., seed=42)