from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from itertools import product
from PIL import Image
import shutil as sh
import numpy as np
import os


def name_generator(init=0, numbers=4):
    index = init
    while True:
        yield str(index).zfill(numbers)
        index += 1


def unify_model(model):
    model_config = model.get_config()
    model_weights = model.get_weights()
    model_config['layers'][0]['config']['batch_input_shape'] = (None, None, None, 3)
    model_config['output_layers'] = []
    met_conv = False
    for ind in reversed(range(len(model_config['layers']))):
        if model_config['layers'][ind]['class_name'] == 'Conv2D':
            model_config['output_layers'].append(
                [model_config['layers'][ind]['name'], 0, 0]
            )
            if not met_conv:
                met_conv=True
            continue
        if not met_conv:
            model_config['layers'].pop(ind)
            model_weights.pop(ind)
    new_model = Model.from_config(model_config)
    new_model.set_weights(model_weights)
    return new_model


def apply_color_mask(image, mask=(1, 1, 1, 1)):
    assert all([m >= 0 and m <= 1 for m in mask]), 'mask should contain [0, 1] values'
    pixels = image.load()
    for i, j in product(range(image.size[0]), range(image.size[1])):
        pixels[i, j] = tuple(round(p * f) for p, f in zip(pixels[i, j], mask))
    return image


def get_filename_template(path_to, symbol, ending):
    def tmp(values):
        string = path_to
        for val in values:
            string += str(val) + symbol
        return string + ending + '/'
    return tmp


def visualize_feature_maps(model, image_paths, path_to, *, clear_to=False):
    new_model = unify_model(model)
    if clear_to and os.listdir(path_to):
        sh.rmtree(path_to, ignore_errors=True)
        os.mkdir(path_to)
    get_pathname = get_filename_template(path_to, '_', 'conv')
    for img_ind, img_path in enumerate(image_paths):
        reference_img = Image.open(img_path)
        reference_img = apply_color_mask(reference_img, mask=(0.2, 0.2, 0.2))
        reference_img = reference_img.convert('RGBA')
        ref_width, ref_height = reference_img.size
        img = load_img(img_path)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = np.divide(img, 255.)
        feature_maps = new_model.predict(img)
        for ind, fmt in enumerate(feature_maps):
            imagename_gen = name_generator()
            current_path = get_pathname([img_ind, len(feature_maps) - ind - 1])
            os.mkdir(current_path)
            width, height, depth = fmt.shape[1:]
            for i in range(depth):
                fm = fmt[0, :, :, i]
                max_val = np.max(fm)
                if max_val > 0:
                    fm = np.divide(fm, max_val)
                fm = np.multiply(fm, 255.)
                img = Image.fromarray(fm)
                img = img.convert('L')
                img = img.resize((ref_width, ref_height))
                if width < ref_width // 2 and height < ref_height // 2:
                    img = img.convert('RGBA')
                    img = Image.blend(reference_img, img, 0.5)
                img.save(current_path + next(imagename_gen) + '_' + str(round(max_val, 2)) + '.png')


if __name__ == '__main__':
    model = load_model('../feature_extractor/model.h5')
    visualize_feature_maps(model, ['../tests/test_0.jpg',
                                   '../tests/test_0.jpg',
                                   '../tests/test_2.jpg',
                                   '../tests/test_3.jpg',
                                   '../tests/test_4.jpg'], '../feature_extractor/new_feature_maps/', clear_to=True)
