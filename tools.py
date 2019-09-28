import pandas as pd


def compute_class_weights(dataframe, *, name_label, count_label):
    mean = dataframe[count_label].sum() / len(dataframe.index)
    return {row[name_label]: mean / row[count_label] for i, row in dataframe.iterrows()}
