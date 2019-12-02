from RPN import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils import plot_model
from keras.models import Model, load_model
import pandas as pd

if __name__ == '__main__':
    model = load_model('../feature_extractor/pretrained_model.h5')
    model = prepare_pretrained_model(model, crop_index=12, lock_index=4, input_shape=(1280, 720, 3))
    rpn = create_rpn_model(model, conv_kernels=96, k=6)

    seed=42
    scales = [1., 1.5]
    sizes = [16, 48, 96]
    anchor_boxes, valid_indices = prepare_rpn_anchor_boxes(
        image_size=(1280, 720),
        feature_map_size=(80, 45),
        sizes=sizes, scales=scales
    )

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
        train_generator, anchor_boxes=anchor_boxes, valid_ab_indices=valid_indices,
        lower_iou_threshold=0.05, upper_iou_threshold=0.4, pos_to_neg_ratio=0.5, seed=seed
    )

    plot_model(rpn, to_file='../rpn/rpn.png', show_shapes=True)
    checkpoints = ModelCheckpoint('../rpn/checkpoints/e{epoch}.h5', period=25)
    best_checkpoint = ModelCheckpoint('../rpn/best_e{epoch}.h5', save_best_only=True)
    csv_logger = CSVLogger('../rpn/log.csv')

    rpn.fit_generator(rpn_train_generator,
                      steps_per_epoch=len(train_generator),
                      epochs=250, verbose=1,
                      callbacks=[csv_logger, checkpoints, best_checkpoint])
