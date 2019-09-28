import pandas as pd
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras_preprocessing.image import ImageDataGenerator
from playsound import playsound
from tools import compute_class_weights
from datetime import datetime


model = Sequential()
model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(80, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(96, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(112, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(192, activation='relu', use_bias=True))
model.add(Dropout(rate=0.25))
model.add(Dense(106, activation='softmax', use_bias=True))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])
print(model.summary())

train = pd.read_csv('train.csv')
train['class_number'] = train['class_number'].astype(str)
train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=train, directory="train/",
    x_col="filename", y_col="class_number",
    class_mode="categorical", target_size=(48, 48),
    batch_size=64, seed=42
)
validation = pd.read_csv('validation.csv')
validation['class_number'] = validation['class_number'].astype(str)
validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=validation, directory="validation/",
    x_col="filename", y_col="class_number",
    class_mode="categorical", target_size=(48, 48),
    batch_size=64, seed=42
)

pd.DataFrame.from_dict(
    {value: key for key, value in train_generator.class_indices.items()},
    columns=['class'], orient='index'
).to_csv('results/labels.csv', index=False)

checkpoint = ModelCheckpoint('results/models/{epoch}.h5', period=25)
tensorboard = TensorBoard(log_dir='results/tb_logs', write_images=True)
logger = CSVLogger('results/' + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.csv')

class_weights = compute_class_weights(
        pd.read_csv('classes_count.csv'), name_label='class_number',
        count_label='count'
)
history = model.fit_generator(
    generator=train_generator,
    validation_data=validation_generator,
    epochs=500, verbose=1, workers=4,
    validation_freq=2, callbacks=[logger, tensorboard, checkpoint],
    class_weight=class_weights
)

model.save('results/complete_model.h5')
playsound('misc/microwave.mp3')
