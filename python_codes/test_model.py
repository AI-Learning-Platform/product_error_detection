import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

model = tf.keras.models.load_model('model/trained_model')
class_names = ['correct', 'wrong']


img = keras.preprocessing.image.load_img(
    './dzgn1.jpeg', target_size=(180, 180)
)
plt.imshow(img)
plt.title('Training and Validation Loss')
plt.show()

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print(
    "Resim {} gözüküyor. {:.2f} ihtimalle."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)