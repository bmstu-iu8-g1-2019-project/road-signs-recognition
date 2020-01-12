import pandas as pd
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from playsound import playsound
from sources.tools import compute_class_weights, cls_wrapper, class_indices, class_weights_array

input = Input(shape=(64, 64, 3), name='Input')
output = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_1')(input)
output = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_2')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_2')(output)
output = Conv2D(filters=48,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_3')(output)
output = Conv2D(filters=48,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_4')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_3')(output)
output = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_5')(output)
output = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_6')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_4')(output)
output = Conv2D(filters=80,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_7')(output)
output = Conv2D(filters=80,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_8')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_5')(output)
output = Conv2D(filters=96,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                kernel_initializer='he_uniform',
                name='Convolutional_9')(output)
output = Flatten()(output)
output = Dropout(rate=0.15, seed=42)(output)
output = Dense(units=200,
               use_bias=True,
               activation='relu',
               name='Dense_1')(output)
output = Dense(units=200,
               use_bias=True,
               activation='relu',
               name='Dense_2')(output)
classification = Dense(units=104,
                       use_bias=True,
                       activation='softmax',
                       name='Classification')(output)
model = Model(inputs=input, outputs=classification)
model.compile(optimizer='adadelta',
              loss=categorical_crossentropy,
              metrics=['accuracy'])
model.summary()
train_data = pd.read_csv('../dataset/train.csv')
train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=train_data, directory='../dataset/train/',
    x_col='filename', y_col='sign_class',
    class_mode='raw', target_size=(64, 64),
    batch_size=64, seed=42
)

validation_data = pd.read_csv('../dataset/validation.csv')
validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
    dataframe=validation_data, directory='../dataset/validation/',
    x_col='filename', y_col='sign_class',
    class_mode='raw', target_size=(64, 64),
    batch_size=64, seed=42
)

classes = pd.read_csv('../dataset/counts.csv')['class']
classes_ind = class_indices(classes.values)

checkpoints = ModelCheckpoint('../results/models/{epoch}.h5', period=25)
best_checkpoint = ModelCheckpoint('../results/models/best.h5', save_best_only=True)
tb_logger = TensorBoard(log_dir='../results/tb_logs/')
csv_logger = CSVLogger('../results/log.csv')
classes_w = compute_class_weights(
    '../dataset/counts.csv',
    name_label='class',
    count_label='count'
)

model.fit_generator(cls_wrapper(train_generator, classes_ind),
                    steps_per_epoch=len(train_generator),
                    validation_data=cls_wrapper(validation_generator, classes_ind),
                    validation_steps=len(validation_generator),
                    class_weight=class_weights_array(classes_ind, classes_w),
                    epochs=500, verbose=1,
                    callbacks=[csv_logger, checkpoints, best_checkpoint, tb_logger])

playsound('../misc/microwave.mp3')
