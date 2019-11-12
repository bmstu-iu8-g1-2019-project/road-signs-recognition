import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from playsound import playsound
from sources.tools import cls_reg_wrapper, class_indices, threadsafe_generator

model = load_model('results/models/250.h5')

test_data = pd.read_csv('dataset/test.csv')
test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=test_data, directory='dataset/test/',
    x_col='filename', y_col=['sign_class', 'width', 'height'],
    class_mode='raw', target_size=(64, 64),
    batch_size=64, seed=42
)

classes = pd.read_csv('dataset/counts.csv')['class']
classes_ind = class_indices(classes.values)
threadsafe_cls_reg_wrapper = threadsafe_generator(cls_reg_wrapper)

print(model.evaluate_generator(generator=threadsafe_cls_reg_wrapper(test_generator, classes_ind),
                               steps=len(test_generator), verbose=1, workers=4))

playsound('misc/microwave.mp3')
