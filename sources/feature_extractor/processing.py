from keras.models import Model


def lock_layers(model, indices):
    """Locks layers on respective indices from indices array"""
    for i in indices:
        model.layers[i].trainable = False
    return model


def prepare_feature_extractor(model,
                              crop_index,
                              lock_index,
                              input_shape):
    """Crops model and locks layers from training
    Manipulates model config since it is only viable option besides rebuilding the whole model
    Receives:
        model - pretrained model
        crop_index - index of last layer in final model
        lock_index - index of last locked layer in final model
        input_shape - since model is convolutional, input could be freely changed
    Returns:
        new cropped model
    """
    config = model.get_config()
    weights = model.get_weights()

    # Edit input layer
    config['layers'][0]['config']['batch_input_shape'] = (None, *input_shape)

    # Crop unnecessary layers
    assert crop_index < len(config['layers']), 'crop_index is out of layers list bounds'
    config['layers'] = config['layers'][:crop_index + 1]

    # Assign new model output
    config['layers'][-1]['config']['name'] = 'FeatureMap'
    config['output_layers'][0][0] = config['layers'][-1]['name']

    # Build cropped model from config and load weights
    model = Model.from_config(config)
    model.set_weights(weights)

    # Lock layers
    assert lock_index < len(config['layers']), 'lock_index is out of cropped layers list bounds'
    return lock_layers(model, list(range(lock_index + 1)))