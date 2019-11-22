from keras.layers import Conv2D, Lambda
from keras.models import Input, Model
from keras.losses import binary_crossentropy, huber_loss
import keras.backend as K
import tensorflow as tf
import numpy as np


def create_rpn_model(input_shape=(None, None, None),
                     ab_per_fm_point=9):
    feature_map = Input(shape=input_shape)
    conv_layer = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same'
    )(feature_map)
    cls = Conv2D(
        filters=ab_per_fm_point,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="RPN_cls"
    )(conv_layer)
    # Выпрямляем тензор в (-1, 1)
    cls = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 1]))(cls)
    reg = Conv2D(
        filters=4 * ab_per_fm_point,
        kernel_size=(1, 1),
        activation="linear",
        kernel_initializer="uniform",
        name="RPN_reg"
    )(conv_layer)
    # Выпрямляем тензор в (-1, 4)
    reg = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, 4]))(reg)
    model = Model(inputs=[feature_map], outputs=[cls, reg])
    model.compile(optimizer='adam', loss={'RPN_cls': binary_crossentropy, 'RPN_reg': huber_loss})
    return model


def cls_loss(target_labels, predicted_label):
    """Обертка для binary_crossentropy. Ошибка для областей, помеченных '-1'
    игнорируется"""
    target_labels = tf.squeeze(target_labels, -1)
    contributing_indices = tf.where(tf.not_equal(target_labels, -1))
    target_labels = tf.gather_nd(target_labels, contributing_indices)
    contributing_prediction = tf.gather_nd(predicted_label, contributing_indices)
    loss = K.binary_crossentropy(target=target_labels,
                                 output=contributing_prediction)
    return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def bbox_loss(target_deltas, target_labels, predicted_deltas):
    """Обертка для huber_loss. Ошибка считается только для областей, помеченных '1'"""
    target_labels = tf.squeeze(target_labels, -1)
    contributing_indices = tf.where(tf.equal(target_labels, 1))
    target_deltas = tf.gather_nd(target_deltas, contributing_indices)
    contributing_prediction = tf.gather_nd(predicted_deltas, contributing_indices)
    loss = huber_loss(target_deltas,
                      contributing_prediction)
    return K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))


def generate_anchor_boxes(sizes, scales, image_size, feature_map_size):
    """Создает все якорные области для текущей конфигурации РПН.
    Принимает:
        sizes [N] - массив размеров якорных областей
        scales [N] - массив отношений сторон (ширина к высоте) якорных областей
        image_size (W, H) - размер изображения
        feature_map_size (fW, fH) - размер входного слоя RPN без его глубины

    Возвращает:
        all_ab [N, x, y, w, h] - все якорные области изображения"""
    image_width, image_height = image_size
    width_stride, height_stride = np.floor(np.divide(image_size, feature_map_size)).astype(int)

    # Организуем координаты всех центров якорных областей в сетку
    # и разложим получившийся тензор в N массивов по два элемента
    x_centers = np.arange(0, image_width, width_stride)
    y_centers = np.arange(0, image_height, height_stride)
    centers = np.dstack(np.meshgrid(x_centers, y_centers)).reshape((-1, 2))

    # Создадем набор размеров ядерных областей
    # Скомбинировать sides и scales в необходимом порядке
    # так же можно с помощью product
    r_sides = np.repeat([sizes], len(scales), axis=1).ravel()
    r_scales = np.repeat([scales], len(sizes), axis=0).ravel()
    ab_pyramid = np.transpose([r_sides * (r_scales ** .5) // 2,
                               r_sides / (r_scales ** .5) // 2]).astype(int)

    # Организуем матрицу из координат центров и соответсвующих им размеров
    # якорных областей. Следующие три строки эквивалентны:
    # [a + b for a, b in product(centers, ab_pyramid)]
    r_centers = np.repeat(centers, len(ab_pyramid), axis=0)
    r_ab_pyramid = np.repeat([ab_pyramid], len(centers), axis=0).reshape((-1, 2))
    return np.hstack(r_centers, r_ab_pyramid)


def valid_anchor_boxes(anchor_boxes, image_size):
    """Возвращает индексы всех якорных областей, которые целиком помещаются
    в изображение
    Принимает:
        anchor_boxes [N, x, y, w, h] - матрица, содержащая все якорные области изображения
        image_size (W, H) - размер изображения
    Возвращает:
        indices [N] - массив индексов, соответсвующих необходимым ядерным областям
        """
    img_width, img_height = image_size
    x, y, width, height = np.transpose(anchor_boxes)

    # Создаем матрицу, хранящую информацию о валидности якорных областей
    indicators = np.array([x - width // 2 >= 0,
                           y - height // 2 >= 0,
                           x + width // 2 <= img_width,
                           y + height // 2 <= img_height]).transpose()

    # Найдем те якорные области, где все параметры оказались валидны
    return np.nonzero(np.all(indicators, axis=0, keepdims=False))


def compute_iou(anchor_boxes, gt_boxes):
    """Вычисяляет IoU между каждой якорной областью и каждой референс областью
    Принимает:
        anchor_boxes [N, x, y, w, h] - якорные области, для которых необходимо считать IoU
        gt_boxes [N, x0, y0, x1, y1] - референс области для которых необходимо считать IoU
    Возвращает:
        ious [len(anchor_boxes)] - максимальные среди всех референс областей
                                   IoU для каждой якорной области
        gt_boxes_index [len(anchor_boxes)] - индексы референс областей среди gt_boxes,
                                             указывающие на область с максимальным IoU
                                             для каждой якорной области"""
    x, y, width, height = np.transpose(anchor_boxes)
    ab_areas = width * height

    # Все якорные области нужно привести к формату с двумя точками в левом
    # верхнем и правом нижнем углу
    w_shift = width // 2
    h_shift = height // 2

    # Для добавления элементов змеиные списки лучше чем np-массивы
    ious = []
    for gt_box in gt_boxes:
        x0, y0, x1, y1 = gt_box
        gt_area = (x1 - x0) * (y1 - y0)

        # Найдем координаты пересечений каждой якорной области с соответствующей
        # референс областью. Площади могут получиться сломанными - не будет выполняться
        # условие x0 < x1 && y0 < y1, но такие области отсеиваются на шаге вычисления области
        x0 = np.maximum(x0, x - w_shift)
        y0 = np.maximum(y0, y - h_shift)
        x1 = np.minimum(x1, x + w_shift)
        y1 = np.minimum(y1, y + h_shift)

        # Вычислим площадь пересечения, отсеивая сломанные области
        int_area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)

        # Вычислим меру IoU
        ious.append(int_area / (ab_areas + gt_area - int_area))
    # Группируем полученные IoU так чтобы каждый ряд представлял IoU одной якорной области со
    # всеми референс областями и затем найдем максимальные IoU для каждой якорной области
    ious = np.transpose(ious)
    gt_index = np.argmax(ious, axis=1)
    ious = np.take(ious, gt_index).ravel()
    return ious, gt_index


def compute_deltas(anchor_boxes, gt_boxes):
    """Вычисляет такие изменения для ядерных областей, чтобы привести их к соответствующим
    референс областям
    Принимает:
        anchor_boxes [N, x, y, w, h] - якорные области
        gt_boxes [N, x0, y0, x1, y1] - соответствующие им референс области
    Возвращает:
        deltas [N, (gt_x_center - x) / w,
                   (gt_y_centet - y) / h,
                   log(gt_width / w),
                   log(gt_height / h)] - необходимые для обучения RPN дельты"""
    x, y, width, height = np.transpose(anchor_boxes)
    x0, y0, x1, y1 = np.transpose(gt_boxes)

    # Дельты удобно считать когда область представлена своим центром и шириной с высотой,
    # поэтому заранее приведем все референс области к такому виду
    gt_width = x1 - x0
    gt_height = y1 - y0
    gt_x_center = x0 + gt_width // 2
    gt_y_center = y0 + gt_height // 2
    return np.array([(gt_x_center - x) / width,
                     (gt_y_center - y) / height,
                     np.log(gt_width / width),
                     np.log(gt_height / height)])


def generate_labels_and_deltas(gt_boxes, anchor_boxes,
                               valid_indices,
                               lower_iou_threshold,
                               upper_iou_threshold,
                               max_positive_samples,
                               max_negative_samples,
                               random_generator):
    """Генерирует метки и дельты для набора референс боксов
    Принимает:
        gt_boxes [N, x0, y0, x1, y1] - пакет референс областей
        anchor_boxes [N, x, y, w, h] - все якорные области изображения
        valid_indices [N] - индексы областей, с которыми будут производиться вычисления
        lower_iou_threshold - верхнее граничное зачение IoU. Любые якорные области
                              с IoU больше этого значения получат метку '1' (foreground)
        upper_iou_threshold - нижнее граничное значение IoU. Любые якорные области
                              с IoU меньше этого значения получают метку '0' (background)
        max_positive_samples - максимальное количество областей, получивших метку '1'
        max_negative_samples - максимальное количество областей, получивших метку '0'
        random_generator - np-генератор случайных чисел
    Возвращает:
        labels [N] - метки всех якорных областей изображения
        deltas [N, dx, dy, dw, dh] - дельты всех якорных областей"""
    # Заранее подготовим массив содержащий IoU для каждой якорной области
    ious = np.zeros(anchor_boxes[0])

    # Вычислим максимальные IoU для необходимых якорных областей и
    # получим индексы соответствующих им референс областей
    ious[valid_indices], gt_boxes_indices = compute_iou(anchor_boxes[valid_indices], gt_boxes)
    labels = np.full(ious.shape, -1)

    # Находим все 'положительные' области
    positive_indices = np.nonzero((ious > upper_iou_threshold)[valid_indices])
    if len(positive_indices) > max_positive_samples:
        positive_indices = random_generator.choice(positive_indices, max_positive_samples, replace=False)
    labels[positive_indices] = 1

    # Находим все 'отрицательные' области
    negative_indices = np.nonzero((ious < lower_iou_threshold)[valid_indices])
    if len(negative_indices) > max_negative_samples:
        negative_indices = random_generator.choice(negative_indices, max_negative_samples, replace=False)
    labels[negative_indices] = 0

    # Вычисляем дельты для всех положительных якорных областей и соответсвующих им референс областей
    gt_boxes = np.take(gt_boxes,
                       gt_boxes_indices[ious[valid_indices] > upper_iou_threshold],
                       axis=0)
    deltas = np.zeros_like(anchor_boxes)
    deltas[positive_indices] = compute_deltas(anchor_boxes[positive_indices],
                                              gt_boxes)

    return labels, deltas


def slow_rpn_generator(pretrained_model,
                       image_data_generator,
                       *,
                       scales, sizes, seed,
                       lower_iou_threshold,
                       upper_iou_threshold,
                       max_positive_samples,
                       max_negative_samples,
                       ):
    """Генерирует необходимые для обучения RPN данные традиционным методом.
    Функция оборачивает генератор изображений и принимает параметры генерируемых областей"""
    # Узнаем размер изображения и карты признаков, поступающей на вход RPN
    # полученные массивы содержат размер пачки по 0 индексу и глубину по 3-му
    image_size = pretrained_model.input_shape[1:3]
    feature_map_size = pretrained_model.output_shape[1:3]

    # Заранее сгененрируем все якорные области
    all_ab = generate_anchor_boxes(sizes, scales,
                                   image_size,
                                   feature_map_size)
    # Получим индексы всех якорных областей, целиком попавших на изображение
    valid_ab_indices = valid_anchor_boxes(all_ab,
                                          image_size)
    random_generator = np.random.Generator(seed)
    for img_batch, gt_boxes_batch in image_data_generator:
        # RPN принимает карты признаков
        feature_maps = pretrained_model.predict(img_batch)
        # TODO(Mocurin)
        labels, deltas = np.apply_along_axis(generate_labels_and_deltas,
                                             gt_boxes_batch, 0,
                                             all_ab, valid_ab_indices,
                                             lower_iou_threshold,
                                             upper_iou_threshold,
                                             max_positive_samples,
                                             max_negative_samples,
                                             random_generator)
        labels = np.squeeze(labels)
        deltas = np.squeeze(deltas)
        yield feature_maps, labels, [deltas, labels]


def greedy_non_maximum_suppression(anchor_boxes, scores, overlap_threshold):
    """Слой NMS, обрабатывающий результаты RPN для уменьшения количества данных,
    попадающих в ROIP
    TODO(Mocurin): Нормально написать что возвращает NMS
    Принимает:
        anchor_boxes [N, x, y, w, h] - якорные области после применения к ним дельт
                                       для эффективной работы стоит предварительно отобрать
                                       те якорные области, которые попали в избражение
        scores [N] - мнение сети по поводу принадлежности области
                     к одному из классов (foreground/background)
        overlap_threshold - максимальное наложение двух областей
    Возвращает:
        picked_anchor_boxes [N, x, y, w, h]
        picked_scores - [N]
            - параметры областей, для которых выполняется условие наложения"""
    x, y, width, height = np.transpose(anchor_boxes)
    areas = width * height
    w_shift = width // 2
    h_shift = height // 2

    # Приведем якорные области в формат левой верхней и правой нижней точек
    x0 = x - w_shift
    y0 = y - h_shift
    x1 = x + w_shift
    y1 = y + h_shift

    sorted_indices = np.argsort(scores.ravel())
    picked_indices = []
    while len(sorted_indices) > 0:
        # Выберем область имеющую нибольший score
        last = sorted_indices[-1]
        sorted_indices = sorted_indices[:-1]
        picked_indices.append(last)

        # Найдем пересечение этой области со всеми остальными якорными областями
        left = np.maximum(x0[last], x0[sorted_indices])
        top = np.maximum(y0[last], y0[sorted_indices])
        right = np.minimum(x1[last], x1[sorted_indices])
        bottom = np.minimum(y1[last], y1[sorted_indices])

        # Отбросим те якорные области, пересечение которых с целевой областью больше заданного отношения
        overlap = np.maximum(0, right - left) * np.maximum(0, top - bottom) / areas[sorted_indices]
        sorted_indices = sorted_indices[overlap.ravel() < overlap_threshold]
    return anchor_boxes[picked_indices], scores[picked_indices]