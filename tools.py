import pandas as pd


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
    result = []
    for i in range(len(classes)):
        result.append([labels.iloc[classes[i], :]['class'], predictions[i][classes[i]]])
    return result
