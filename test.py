import pandas as pd
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from playsound import playsound

model = load_model('results/models/150.h5')
t_datagen = ImageDataGenerator(rescale=1./255)
test = pd.read_csv('test.csv')
test['class_number'] = test['class_number'].astype(str)
test_generator = t_datagen.flow_from_dataframe(
    dataframe=test, directory="test/",
    x_col="filename", y_col="class_number",
    class_mode="categorical", target_size=(48, 48),
    batch_size=32, shuffle=False
)
print(model.evaluate_generator(generator=test_generator, verbose=1, workers=4))
prediction = pd.read_csv('predict.csv')
prediction['class_number'] = prediction['class_number'].astype(str)
t_datagen = ImageDataGenerator(rescale=1./255)
prediction_generator = t_datagen.flow_from_dataframe(
    dataframe=prediction, directory="predict/",
    x_col="filename", y_col="class_number",
    class_mode="categorical", target_size=(48, 48),
    batch_size=1, shuffle=False
)
predictions = model.predict_generator(generator=prediction_generator, verbose=1, workers=4)
classes = predictions.argmax(axis=-1)
labels = pd.read_csv('results/labels.csv')
result = []
for i in range(len(classes)):
    result.append([labels.iloc[classes[i],:]['class'], predictions[i][classes[i]]])
print(result)

playsound('misc/microwave.mp3')
