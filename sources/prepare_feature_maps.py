import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.preprocessing.image import load_img, img_to_array
from os.path import splitext
from RPN import prepare_pretrained_model


def images_batch_generator(directory, filenames, batch_size=16, image_size=None):
    """Generator which yields images form directory and respective filenames"""
    n_images = len(filenames)
    previous = 0
    if image_size is not None:
        # Image size fore keras img_load requires (height, width)
        image_size = (image_size[1], image_size[0])
    # We need to generate a number beyond size of "paths" to make a last slice
    for current in range(batch_size, n_images + batch_size, batch_size):
        filenames_batch = filenames[previous:current]
        previous = current
        paths_batch = [directory + fname for fname in filenames_batch]
        images_batch = [load_img(path, target_size=image_size) for path in paths_batch]
        images_batch = [img_to_array(img) for img in images_batch]
        images_batch = [np.transpose(img, [1, 0, 2]) for img in images_batch]
        images_batch = np.array([np.divide(img, 255.) for img in images_batch])
        yield images_batch, filenames_batch


def compute_fmaps(model, img_filenames, img_directory, fm_directory, batch_size=16, image_size=None):
    """Takes model and saves feature maps, computed by it, to 'fm_directory'"""
    gen = images_batch_generator(img_directory, img_filenames, batch_size, image_size)
    new_filenames = []
    n_samples = len(img_filenames)
    current_sample = 0
    for images_batch, filenames_batch in gen:
        feature_maps = model.predict_on_batch(images_batch)
        filenames_batch = [splitext(fname)[0] + '.npy' for fname in filenames_batch]
        new_filenames += filenames_batch
        for fm, fname in zip(feature_maps, filenames_batch):
            np.save(fm_directory + fname, fm)
            current_sample += 1
            print(fname + ' saved! ' + '{}/{} image'.format(current_sample, n_samples)
                  + ' {} Mb'.format(round(fm.nbytes / 1024 / 1024, 2)))
    return new_filenames


if __name__ == '__main__':
    model = load_model('../feature_extractor/pretrained_model.h5')
    pretrained_model = prepare_pretrained_model(model, crop_index=12)
    dataset = pd.read_json('../dataset/dataset.json')
    dataset['filename'] = compute_fmaps(pretrained_model,
                                        dataset['filename'].values,
                                        '../dataset/rtsd-frames/',
                                        '../dataset/fm-dataset/',
                                        batch_size=8)
    dataset.to_json('../dataset/fm_dataset.json')


