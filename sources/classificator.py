import pandas as pd
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.losses import huber_loss, categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from playsound import playsound
from sources.tools import compute_class_weights, cls_reg_wrapper, class_indices, class_weights_array, threadsafe_generator

input = Input(shape=(64, 64, 3))
layers = Conv2D(filters=32,
                kernel_size=3,
                padding='same',
                activation='relu')(input)
layers = MaxPooling2D(pool_size=2)(layers)
layers = Conv2D(filters=48,
                kernel_size=3,
                padding='same',
                activation='relu')(layers)
layers = MaxPooling2D(pool_size=2)(layers)
layers = Conv2D(filters=64,
                kernel_size=3,
                padding='same',
                activation='relu')(layers)
layers = MaxPooling2D(pool_size=2)(layers)
layers = Conv2D(filters=80,
                kernel_size=3,
                padding='same',
                activation='relu')(layers)
layers = MaxPooling2D(pool_size=2)(layers)
layers = Conv2D(filters=96,
                kernel_size=3,
                padding='same',
                activation='relu')(layers)
layers = Flatten()(layers)
reg = Dense(units=2, activation='linear', name='lazy_reg')(layers)
layers = Dropout(rate=0.2)(layers)
cls = Dense(units=114, activation='softmax', name='lazy_cls')(layers)
model = Model(inputs=input, outputs=[cls, reg])
model.compile(optimizer='adam',
              loss={'lazy_cls': categorical_crossentropy, 'lazy_reg': huber_loss},
              metrics={'lazy_cls': 'accuracy', 'lazy_reg': 'accuracy'})

train_data = pd.read_csv('dataset/train.csv')
train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=train_data, directory='dataset/train/',
    x_col='filename', y_col=['sign_class', 'width', 'height'],
    class_mode='raw', target_size=(64, 64),
    batch_size=64, seed=42
)

validation_data = pd.read_csv('dataset/validation.csv')
validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=validation_data, directory='dataset/validation/',
    x_col='filename', y_col=['sign_class', 'width', 'height'],
    class_mode='raw', target_size=(64, 64),
    batch_size=64, seed=42
)

classes = pd.read_csv('dataset/counts.csv')['class']
classes_ind = class_indices(classes.values)

checkpoints = ModelCheckpoint('results/models/{epoch}.h5', period=25)
tb_logger = TensorBoard(log_dir='results/tb_logs/')
csv_logger = CSVLogger('results/log.csv')
classes_w = compute_class_weights(
    'dataset/counts.csv',
    name_label='class',
    count_label='count'
)

threadsafe_cls_reg_wrapper = threadsafe_generator(cls_reg_wrapper)

model.fit_generator(threadsafe_cls_reg_wrapper(train_generator, classes_ind),
                    steps_per_epoch=len(train_generator),
                    validation_data=threadsafe_cls_reg_wrapper(validation_generator, classes_ind),
                    validation_steps=len(validation_generator),
                    class_weight={'lazy_cls': class_weights_array(classes_ind, classes_w)},
                    epochs=500, verbose=1, workers=4,
                    callbacks=[csv_logger, checkpoints, tb_logger])

model.save('results/models/final.h5')
playsound('misc/microwave.mp3')
