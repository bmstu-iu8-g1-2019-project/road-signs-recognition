import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from playsound import playsound
from tools import get_predictions_with_prob

model = load_model('results/models/150.h5')

test = pd.read_csv('dataset/test.csv')
test['class_number'] = test['class_number'].astype(str)
test_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=test, directory="dataset/test/",
    x_col="filename", y_col="class_number",
    class_mode="categorical", target_size=(48, 48),
    batch_size=32, shuffle=False
)
print(model.evaluate_generator(generator=test_generator, verbose=1, workers=4))

prediction = pd.read_csv('dataset/predict.csv')
prediction['class_number'] = prediction['class_number'].astype(str)
prediction_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    dataframe=prediction, directory="dataset/predict/",
    x_col="filename", y_col="class_number",
    class_mode="categorical", target_size=(48, 48),
    batch_size=1, shuffle=False
)

predictions = model.predict_generator(generator=prediction_generator, verbose=1, workers=4)
print(get_predictions_with_prob('dataset/labels.csv', predictions))

playsound('misc/microwave.mp3')
