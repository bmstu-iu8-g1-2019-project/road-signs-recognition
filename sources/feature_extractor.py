from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.models import Model
from layers import residual_block

input = Input(shape=(960, 540, 3))
output = Conv2D(
    filters=32,
    kernel_size=5,
    padding='same',
    activation='relu'
)(Input)
output = MaxPooling2D(
    pool_size=2)(output)
output = residual_block(
    output, filters=32)
output = residual_block(
    output, filters=64, block_stride=2)
output = residual_block(
    output, filters=64)
output = residual_block(
    output, filters=64)
output = residual_block(
    output, filters=128, block_stride=2)
output = residual_block(
    output, filters=128)
output = residual_block(
    output, filters=128)
output = AveragePooling2D(
    pool_size=2)(output)
output = Flatten()(output)
output = Dropout(
    rate=0.25)(output)
output = Dense(
    units=106, activation='sigmoid')(output)
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# TODO(Mocurin): Add data generators and fitting
