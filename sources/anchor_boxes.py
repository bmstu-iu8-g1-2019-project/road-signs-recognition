import numpy as np


ab_parameters = {
    'img_size': (1280, 720),
    'fm_size': (80, 45),
    'ss_ratios': (16, 16),
    'ab_sizes': [128],
    'ab_ratios': [1]
}


def center_to_corners(box):
    """Преобразует прямоугольную область
    из формата center в формат corners"""
    x, y, width, height = box
    wshift = width // 2
    hshift = height // 2
    return [x - wshift, y - hshift, x + wshift, y + hshift]


def center_to_upleft(box):
    """Преобразует прямоугольную область
    из формата center в формат upleft"""
    x, y, width, height = box
    wshift = width // 2
    hshift = height // 2
    return [x - wshift, y - hshift, width, height]


def corners_to_center(box):
    """Преобразует прямоугольную область
    из формата corners в формат center"""
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    return [left + width // 2, top + height // 2, width, height]


def upleft_to_center(box):
    """Преобразует прямоугольную область
    из формата upleft в формат center"""
    x, y, width, height = box
    wshift = width // 2
    hshift = height // 2
    return [x + wshift, y + hshift, width, height]


def box_area(box):
    """Вычисляет площадь прямоугольной области.
    Box необходимо предавать в формате corners
    Если область построена неправильно (например left > right),
    то будет возвращен 0"""
    x0, y0, x1, y1 = box
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def box_intersection(first_box, second_box):
    """Возвращает новую область, которая является пересечением
    двух областейна входе. Области необходимо передавать
    в формате corners
    В случае, если области не перескаются,
    то будет возвращна область с left > right || top > bottom,
    поэтому области необходимо проверять"""
    x0, y0, x1, y1 = first_box
    x2, y2, x3, y3 = second_box
    return [max(x0, x2),
            max(y0, y2),
            min(x1, x3),
            min(y1, y3)]


def compute_iou(first_box, second_box):
    """Вычисляет IoU-меру между двумя прямоугольными оластями
    *IoU - intersection over union - отношение площади пересечения
    двух областей к площади их объединения"""
    intersection = box_intersection(first_box, second_box)
    intersection_area = box_area(intersection)
    first_box_area = box_area(first_box)
    second_box_area = box_area(second_box)
    return intersection_area / float(
        first_box_area + second_box_area - intersection_area
    )


def gen_anchor_boxes(gt_box):
    """Создает набор всех якорных областей, хоть сколько-нибудь покрывающих данную область
    Принимает gt_box - ground-truth box - область, которую будут пересекать якорные области,
    причем сторого в формате center
    Возвращает индексы якорных областей, параметры самих якорных областей
    и IoU меру полученных областей с gt_box"""
    ngt_box = center_to_corners(gt_box)
    left, top, right, bottom = ngt_box
    ab_sizes = ab_parameters['ab_sizes']
    ab_ratios = ab_parameters['ab_ratios']
    img_width, img_height = ab_parameters['img_size']
    w_stride, h_stride = ab_parameters['ss_ratios']

    res_indices = []
    res_ab = []
    res_iou = []
    for i in range(len(ab_sizes)):
        for j in range(len(ab_ratios)):
            w_shift = ab_sizes[i] * (ab_ratios[j] ** .5) // 2
            h_shift = ab_sizes[i] / (ab_ratios[j] ** .5) // 2

            # Определяем область, содержащую центры якорных областей, пересекающихся
            # с gt_box и имеющих параметры ab_sizes[i] и ab_ratios[j]
            left_border = left - w_shift
            top_border = top - h_shift
            right_border = right + w_shift
            bottom_border = bottom + h_shift

            # Определяем какие центры якорных областей попали внутрь области
            # и исключаем те, области которых выходят за границы изображения
            left_index = max(np.ceil((left_border - 1) / w_stride),
                             np.ceil((w_shift - 1) / w_stride))
            right_index = min(np.floor((right_border - 1) / w_stride),
                              np.floor((img_width - w_shift - 1) / w_stride))
            if left_index > right_index:
                continue

            top_index = max(np.ceil((top_border - 1) / h_stride),
                            np.ceil((h_shift - 1) / h_stride))
            bottom_index = min(np.floor((bottom_border - 1) / h_stride),
                               np.floor((img_height - h_shift - 1) / h_stride))
            if top_index > bottom_index:
                continue

            # Создаем сетку из индексов, выпрямляем ее в матрицу (-1, 2)
            # Из получившейся матрицы находим координаты центров всех якорных областей
            # И затем группируем полученные данные
            x = np.arange(left_index, right_index + 1)
            y = np.arange(top_index, bottom_index + 1)
            grid = np.meshgrid(x, y)
            paired = np.dstack(grid)
            paired = paired.reshape((-1, 2))
            ab_indices = np.hstack((paired,
                                    [[i + j]] * len(paired))
                                   ).astype(int)
            res_indices += ab_indices.tolist()
            ab_chosen = np.hstack((np.reshape(paired[:, 0] * w_stride + 1, (-1, 1)),
                                   np.reshape(paired[:, 1] * h_stride + 1, (-1, 1)),
                                   [[w_shift * 2]] * len(ab_indices),
                                   [[h_shift * 2]] * len(ab_indices))
                                  ).astype(int)
            res_ab += ab_chosen.tolist()
            ab_iou = np.reshape(
                [compute_iou(center_to_corners(ab), ngt_box)
                 for ab in ab_chosen], (-1, 1))
            res_iou += ab_iou.tolist()
        return res_indices, res_ab, res_iou
