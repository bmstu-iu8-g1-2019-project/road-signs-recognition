from keras.layers import Conv2D, LeakyReLU, BatchNormalization, add


def residual_block(layer, *, filters, first_kernel=3, second_kernel=3, block_stride=1):
    shortcut = layer
    layer = Conv2D(filters, kernel_size=first_kernel, stride=block_stride, padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)
    layer = Conv2D(filters, kernel_size=second_kernel, padding='same')(layer)
    layer = BatchNormalization()(layer)
    if block_stride > 1:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=block_stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    layer = add([shortcut, layer])
    layer = LeakyReLU()(layer)
    return layer




