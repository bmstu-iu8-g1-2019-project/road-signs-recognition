from rpn.generation import rpn_generator, RPNconfig
from rpn.rpn import make_cls_wrapper, make_reg_wrapper, training_rpn_model, ClsMetricWrapper, RegMetricWrapper
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.metrics import BinaryAccuracy, MeanAbsoluteError, FalsePositives, TruePositives
from feature_extractor.processing import prepare_feature_extractor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import plot_model
from keras.models import load_model
from tools import versionize
import pandas as pd


# TODO(Mocurin) Collect ALL configs


if __name__ == '__main__':
    seed = 42
    rpn_config = RPNconfig(image_size=(1280, 720),
                           fm_size=(80, 45),
                           sizes=[24, 48, 96],
                           scales=[0.5, 1., 1.5])

    model = load_model('../feature_extractor/versions/FE_v0/9conv.h5')
    feature_extractor = prepare_feature_extractor(model, crop_index=13, lock_index=3, input_shape=(1280, 720, 3))
    rpn = training_rpn_model(feature_extractor,
                             anchors_per_loc=rpn_config.anchors_per_fm_point,
                             seed=seed)
    cls_loss = make_cls_wrapper(BinaryCrossentropy())
    reg_loss = make_reg_wrapper(Huber())
    cls_acc = ClsMetricWrapper(BinaryAccuracy(), name='acc')
    reg_mae = RegMetricWrapper(MeanAbsoluteError(), name='mae')
    rpn.compile(optimizer='adadelta',
                loss={'bbox_reg': reg_loss,
                      'bbox_cls': cls_loss},
                metrics={'bbox_reg': reg_mae,
                         'bbox_cls': cls_acc})

    version_dir, version_insides = versionize('versions/', root='RPN')
    version_config_dir, version_weights_dir = version_insides

    rpn_config.save_json(version_dir + 'rpn_config.json')
    plot_model(rpn, to_file=version_dir + 'graph.png', show_shapes=True)
    best_config = ModelCheckpoint(version_config_dir + 'best.h5',
                                  save_best_only=True, monitor='bbox_cls_acc')
    best_weights = ModelCheckpoint(version_weights_dir + 'best.h5',
                                   save_best_only=True, monitor='bbox_cls_acc', save_weights_only=True)
    csv_logger = CSVLogger(version_dir + 'log.csv')

    train_data = pd.read_json('../dataset/train.json')
    train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        dataframe=train_data,
        directory='../dataset/train/',
        x_col='filename', y_col='gt_boxes',
        batch_size=8, class_mode='raw',
        seed=seed, target_size=(1280, 720),
        interpolation='lanczos'
    )
    rpn_train_generator = rpn_generator(
        train_generator, 'varying', rpn_config.anchor_boxes, rpn_config.valid_indices,
        lower_threshold=0.05, upper_threshold=0.4, ratio=.5, seed=seed
    )

    validation_data = pd.read_json('../dataset/validation.json')
    validation_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        dataframe=validation_data,
        directory='../dataset/validation/',
        x_col='filename', y_col='gt_boxes',
        batch_size=8, class_mode='raw',
        seed=seed, target_size=(1280, 720),
        interpolation='lanczos'
    )
    rpn_validation_generator = rpn_generator(
        validation_generator, 'varying', rpn_config.anchor_boxes, rpn_config.valid_indices,
        lower_threshold=0.05, upper_threshold=0.4, ratio=.5, seed=seed
    )

    rpn.fit_generator(
        generator=rpn_train_generator,
        steps_per_epoch=500,
        validation_data=rpn_validation_generator,
        validation_steps=125,
        epochs=250, verbose=1,
        callbacks=[csv_logger,
                   best_config,
                   best_weights]
    )