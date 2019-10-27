from keras.layers import Conv2D
from keras.models import Input, Model
from keras.losses import binary_crossentropy, huber_loss

k = 9

feature_map = Input(shape=(None, None, 128))

conv_layer = Conv2D(
    filters=128,  # TODO(Mocurin): Change to feature map depth
    kernel_size=(3, 3),
    padding='same'
)(feature_map)

cls = Conv2D(
    filters=1 * k,
    kernel_size=(1, 1),
    activation="softmax",
    kernel_initializer="uniform",
    name="RPN_cls"
)(conv_layer)

reg = Conv2D(
    filters=4 * k,
    kernel_size=(1, 1),
    activation="linear",
    kernel_initializer="uniform",
    name="RPN_reg"
)(conv_layer)

model = Model(inputs=[feature_map], outputs=[cls, reg])
model.compile(optimizer='adam', loss={'RPN_cls': binary_crossentropy, 'RPN_reg': huber_loss})
