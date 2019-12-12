from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy, Huber
from tensorflow.keras.metrics import BinaryAccuracy, MeanAbsoluteError
from rpn.generation import rpn_generator, RPNconfig
from rpn.rpn import make_cls_wrapper, make_reg_wrapper, ThresholdedRegularizer, ClsMetricWrapper, RegMetricWrapper
import pandas as pd


if __name__ == '__main__':
    seed = 42
    rpn_config = RPNconfig.load_json('versions/RPN_v8/rpn_config.json')

    cls_loss = make_cls_wrapper(BinaryCrossentropy(from_logits=True))
    reg_loss = make_reg_wrapper(Huber())
    cls_acc = ClsMetricWrapper(BinaryAccuracy(), name='acc')
    reg_mae = RegMetricWrapper(MeanAbsoluteError(), name='mae')
    rpn = load_model('versions/RPN_v8/configs/best.h5',
                     custom_objects={'ThresholdedRegularizer': ThresholdedRegularizer,
                                     'reg_processer': reg_loss,
                                     'cls_processer': cls_loss})
    rpn.compile(optimizer=rpn.optimizer,
                loss=rpn.loss,
                metrics={'bbox_reg': reg_mae, 'bbox_cls_log': cls_acc})

    test_data = pd.read_json('../dataset/test.json')
    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        dataframe=test_data,
        directory='../dataset/test/',
        x_col='filename', y_col='gt_boxes',
        batch_size=8, class_mode='raw',
        seed=seed, target_size=(1280, 720),
        interpolation='lanczos'
    )
    rpn_test_generator = rpn_generator(
        test_generator, 'varying', rpn_config.anchor_boxes, rpn_config.valid_indices,
        lower_threshold=0.05, upper_threshold=0.4, ratio=.5, seed=seed
    )

    result = rpn.evaluate_generator(rpn_test_generator,
                                    steps=len(test_generator),
                                    verbose=1)
    print('Evaluation result:')
    print(rpn.metrics_names)
    print(result)