from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from itertools import product
from PIL import Image, ImageDraw, ImageFont
import shutil as sh
import numpy as np
import os


def name_generator(init=0, numbers=4, img_format='.png'):
    index = init
    while True:
        yield str(index).zfill(numbers) + img_format
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
    def tmp(img_ind, map_ind):
        return path_to + str(img_ind) + symbol + \
               'image/' + str(map_ind) + \
               symbol + ending + '/'
    return tmp


def get_legend_template(font_path, *, font_size=14, position=(0, 0), color=(0, 0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    def tmp(image, text):
        draw = ImageDraw.Draw(image)
        draw.text(position, text, color, font=font)
        return draw
    return tmp


def get_legend_text(value, depth, conv):
    return 'Max Activation: {} \n' \
           'Feature map layer index: {} \n' \
           'Convolutional layer index: {} \n' \
        .format(str(value), str(depth), str(conv))


def visualize_feature_maps(model, image_paths, path_to, *, clear_to=False):
    new_model = unify_model(model)
    if clear_to and os.listdir(path_to):
        sh.rmtree(path_to, ignore_errors=True)
        os.mkdir(path_to)
    get_pathname = get_filename_template(path_to, '_', 'conv')
    draw_legend = get_legend_template('../misc/calibril.ttf', color=(255, 0, 0))
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
            conv_ind = len(feature_maps) - ind - 1
            current_path = get_pathname(img_ind, conv_ind)
            os.makedirs(current_path)
            width, height, depth = fmt.shape[1:]
            for depth_ind in range(depth):
                fm = fmt[0, :, :, depth_ind]
                max_val = np.max(fm)
                if max_val > 0:
                    fm = np.divide(fm, max_val)
                fm = np.multiply(fm, 255.)
                img = Image.fromarray(fm)
                # Intentional
                img = img.convert('L')
                img = img.resize((ref_width, ref_height))
                img = img.convert('RGBA')
                if width < ref_width // 2 and height < ref_height // 2:
                    img = Image.blend(reference_img, img, 0.5)
                legend = get_legend_text(max_val, depth_ind, conv_ind)
                draw_legend(img, legend)
                img.save(current_path + next(imagename_gen))


def arrange_images_as_grid(images, grid_size):
    g_width, g_height = grid_size
    assert g_width * g_height == len(images), 'Can not place images in such grid'
    i_sizes = [img.size for img in images]
    i_width = sum([size[0] for size in i_sizes]) // len(images)
    i_height = sum([size[1] for size in i_sizes]) // len(images)
    canvas = Image.new('RGBA', (i_width * g_width, i_height * g_height))
    for i, j in product(range(g_width), range(g_height)):
        img = images[i + j * g_width]
        img = img.resize((i_width, i_height), Image.ANTIALIAS)
        canvas.paste(img, (i * i_width, j * i_height))
    return canvas


def collect_and_arrange_images(images_path, conv_targets, path_to, *, grid_size=(2, 2), clear_to=False):
    if clear_to and os.listdir(path_to):
        sh.rmtree(path_to, ignore_errors=True)
        os.mkdir(path_to)
    for tg in conv_targets:
        conv_paths = [path + tg for path in images_path]
        image_lists = [os.listdir(path) for path in conv_paths]
        full_paths = []
        for path, names in zip(conv_paths, image_lists):
            full_paths.append([path + name for name in names])
        full_paths = np.array(full_paths).T
        save_path = path_to + tg
        os.mkdir(save_path)
        name_gen = name_generator()
        for paths in full_paths:
            images = [Image.open(path) for path in paths]
            grid = arrange_images_as_grid(images, grid_size=grid_size)
            grid.save(save_path + next(name_gen))


if __name__ == '__main__':
    model = load_model('../feature_extractor/model.h5')
    collect_and_arrange_images(['../feature_extractor/new_feature_maps/0_image/',
                                '../feature_extractor/new_feature_maps/1_image/',
                                '../feature_extractor/new_feature_maps/2_image/',
                                '../feature_extractor/new_feature_maps/3_image/'],
                               ['2_conv/',
                                '3_conv/',
                                '4_conv/'],
                               '../feature_extractor/combined/',
                               clear_to=True)
    # visualize_feature_maps(model, ['../tests/test_0.jpg',
    #                                '../tests/test_2.jpg',
    #                                '../tests/test_3.jpg',
    #                                '../tests/test_4.jpg'], '../feature_extractor/new_feature_maps/', clear_to=True)