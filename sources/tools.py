import pandas as pd
import numpy as np
import threading


def compute_class_weights(filename, *, name_label, count_label):
    df = pd.read_csv(filename)
    mean = df[count_label].sum() / len(df.index)
    return {row[name_label]: mean / row[count_label] for i, row in df.iterrows()}


def save_generator_labels(filename, classes):
    pd.DataFrame.from_dict(
        {value: key for key, value in classes.items()},
        columns=['class'], orient='index'
    ).to_csv(filename, index=False)


def get_predictions_with_prob(filename, predictions):
    classes = predictions.argmax(axis=-1)
    labels = pd.read_csv(filename)
    result = {}
    for i in range(len(classes)):
        result[i] = {'prediction': labels.iloc[classes[i], :]['class'],
                     'probability': predictions[i][classes[i]]}
    return result


def class_indices(classes_arr):
    """Returns a dict with classes mapped to indices"""
    size = len(classes_arr)
    return {classes_arr[i]: i for i in range(size)}


def indicator_vector(size, indices_arr):
    """Creates a numpy array with ones on given positions
    and zeroes on others"""
    arr = np.zeros(size)
    arr[indices_arr] = 1
    return arr


def cls_reg_wrapper(generator, cls_indices_dict):
    """Simply arranges an output of generator to match a model
    with a [cls, reg] output"""
    size = len(cls_indices_dict)
    for img, row in generator:
        arr = np.array(row)
        cls = arr[:, 0].ravel()
        cls = np.array([indicator_vector(size, cls_indices_dict[name]) for name in cls])
        reg = arr[:, [1, 2]]
        yield img, [cls, reg]


def class_weights_array(cls_indices_dict, cls_weights_dict):
    """Returns an array containing weights for every class according
    to its index. Receives a class mapping function and a dictionary with
    classes mapped to weights"""
    arr = np.zeros(len(cls_indices_dict))
    for key, value in cls_weights_dict.items():
        ind = cls_indices_dict[key]
        arr[ind] = value
    return arr


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

