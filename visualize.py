from keras.models import load_model
from keras.utils import plot_model
from playsound import playsound

model = load_model('results/models/150.h5')
print(model.summary())
plot_model(model, to_file='results/report/A2_500.png', show_shapes=True)
playsound('misc/microwave.mp3')