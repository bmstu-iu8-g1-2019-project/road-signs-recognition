import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from playsound import playsound
from sources.tools import cls_wrapper, class_indices

model = load_model('../results/models/best.h5')

test_data = pd.read_csv('../dataset/test.csv')
test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=test_data, directory='../dataset/test/',
    x_col='filename', y_col='sign_class',
    class_mode='raw', target_size=(64, 64),
    batch_size=64, seed=42
)

classes = pd.read_csv('../dataset/counts.csv')['class']
classes_ind = class_indices(classes.values)

print(model.evaluate_generator(generator=cls_wrapper(test_generator, classes_ind),
                               steps=len(test_generator), verbose=1, workers=4))
playsound('misc/microwave.mp3')
