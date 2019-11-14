from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model


input = Input(shape=(64, 64, 3), name='Input')
output = Conv2D(filters=32,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                name='Convolutional_1')(input)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_1')(output)
output = Conv2D(filters=48,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                name='Convolutional_2')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_2')(output)
output = Conv2D(filters=64,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                name='Convolutional_3')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_3')(output)
output = Conv2D(filters=80,
                kernel_size=(3, 3),
                kernel_initializer='uniform',
                padding='same',
                use_bias=True,
                activation='relu',
                name='Convolutional_4')(output)
output = MaxPooling2D(pool_size=(2, 2),
                      name='Max_Pooling_4')(output)
output = Conv2D(filters=96,
                kernel_size=(3, 3),
                padding='same',
                use_bias=True,
                activation='relu',
                name='Convolutional_5')(output)
output = Dropout(rate=0.15,
                 name='Tensor_Dropout')(output)
output = Conv2D(filters=32,
                kernel_size=(1, 1),
                use_bias=True,
                activation='relu',
                name='Compression_Layer')(output)
output = Flatten()(output)
lazy_reg = Dense(units=2,
                 use_bias=True,
                 activation='sigmoid',
                 name='Lazy_Reg')(output)
lazy_cls = Dense(units=114,
                 use_bias=True,
                 activation='softmax',
                 name='Lazy_Cls')(output)
model = Model(inputs=[input],
              outputs=[lazy_reg, lazy_cls])


