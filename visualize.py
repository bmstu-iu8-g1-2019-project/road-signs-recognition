from keras.models import load_model
from keras.utils import plot_model

model = load_model('results/models/150.h5')
plot_model(model, to_file='results/report/A1_350.png', show_shapes=True)
