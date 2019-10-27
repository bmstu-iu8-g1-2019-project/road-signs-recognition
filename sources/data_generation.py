from sources.tools import region_area, region_intersection
from itertools import permutations, takewhile, dropwhile
from PIL import Image
import pandas as pd
import numpy as np
import random as rd


# computations are easier in two points format
def two_points(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


# dataset, tho is in upper left format
def upper_left(box):
    x_left, y_top, x_right, y_bottom = box
    return [x_left, y_top, x_right - x_left, y_bottom - y_top]


def reformat(box, box_type):
    if box_type == 'two_points':
        return upper_left(box)
    if box_type == 'upper_left':
        return two_points(box)
    raise NameError('Unknown box format')


# align should be applied after reformatting to upper left
def align_box(box, region):
    xr, yr, wr, hr = region
    x, y, w, h = box
    assert all((x >= xr,
                y >= yr,
                x + w <= xr + wr,
                y + h <= yr + hr)), 'Box does not belong to region'
    return [x - xr, y - yr, w, h]


def choose_point(region):
    x_left, y_top, x_right, y_bottom = region
    return rd.randrange(x_left, x_right + 1), rd.randrange(y_top, y_bottom + 1)


def inverse_indices(length, exc_indices):
    return list(dropwhile(lambda i: i in exc_indices, range(length)))


def is_valid_region(region):
    x_left, y_top, x_right, y_bottom = region
    return x_left <= x_right and y_top <= y_bottom


def is_subset(first_list, second_list):
    return all(elem in second_list for elem in first_list)


def specify_region_proposal(boxes, patch_size, image_size):
    if not len(boxes):
        return []
    pwidth, pheight = patch_size
    iwidth, iheight = image_size
    x_left, y_top, x_right, y_bottom = np.transpose(boxes)
    return [max(1, max(x_right) - pwidth),
            max(1, max(y_bottom) - pheight),
            min(iwidth - pwidth + 1, min(x_left)),
            min(iheight - pheight + 1, min(y_top))]


def subtract_regions(include_region, exclude_region):
    if not len(exclude_region):
        return [include_region]
    intersection_region = region_intersection(include_region, exclude_region)
    if region_area(intersection_region) == 0:
        return [include_region]
    x1, y1, x4, y4 = include_region
    x2, y2, x3, y3 = intersection_region
    # create predefined regions and then exclude redundant
    result_regions = [[x1, y1, x2, y4],  # left
                      [x2, y1, x3, y2],  # top
                      [x3, y1, x4, y4],  # right
                      [x2, y3, x3, y4]]  # bottom
    return list(takewhile(lambda region: region_area(region) > 0, result_regions))


def split_image(dataframe, *, image_size, patch_size):
    boxes = dataframe.iloc[:, 1:5].values
    labels = dataframe.iloc[:, 5:].values
    boxes = np.apply_along_axis(two_points, axis=1, arr=boxes)
    result_data = []
    taken_indices = []
    for n in range(0, len(boxes) // 2 + 1):
        for include_indices in permutations(range(len(boxes)), len(boxes) - n):
            if is_subset(include_indices, taken_indices):
                continue
            exclude_indices = inverse_indices(len(boxes), include_indices)
            include_region = specify_region_proposal(boxes.take(include_indices, axis=0),
                                                     patch_size, image_size)
            exclude_region = specify_region_proposal(boxes.take(exclude_indices, axis=0),
                                                     patch_size, image_size)
            if any([include_region and not is_valid_region(include_region),
                    exclude_region and not is_valid_region(exclude_region)]):
                continue
            taken_indices = np.unique([*include_indices, *taken_indices])
            final_regions = subtract_regions(include_region, exclude_region)
            region_root = choose_point(rd.choice(final_regions))
            result_data.append([region_root,
                                boxes.take(include_indices, axis=0),
                                labels.take(include_indices, axis=0)])
    return result_data


def image_name_generator(init=0, numbers=7, image_format='.jpg'):
    index = init
    while True:
        yield str(index).zfill(numbers) + image_format
        index += 1


def split_images_dataframe(filename_from, filename_to, *,
                           input_path, output_path, target_size):
    df = pd.read_csv(filename_from)
    df_columns = df.columns
    df_grouped = df.groupby('filename')
    df_filenames = df.filename.unique()
    df_result = pd.DataFrame(columns=df_columns)
    name_generator = image_name_generator()
    for filename in df_filenames:
        group = df_grouped.get_group(filename)
        image = Image.open(input_path + filename)
        tmp_result = split_image(group,
                                 image_size=image.size,
                                 patch_size=target_size)
        for root, boxes, labels in tmp_result:
            region = [*root, *target_size]
            left, top = root
            right, bottom = np.add(root, target_size)
            cropped_image = image.crop((left, top, right, bottom))
            name = next(name_generator)
            cropped_image.save(output_path + name)
            boxes = np.apply_along_axis(upper_left,
                                        axis=1,
                                        arr=boxes)
            boxes = np.apply_along_axis(align_box,
                                        axis=1,
                                        arr=boxes,
                                        region=region)
            names_column = [[name]] * len(boxes)
            values = np.hstack((names_column, boxes, labels))
            df_tmp = pd.DataFrame(values, columns=df_columns)
            df_result = df_result.append(df_tmp, ignore_index=True)
    df_result.to_csv(filename_to, index=False)
