import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

model = tf.keras.models.load_model('saved_model/my_model')
class_names = ['hatasız', 'hatalı']


img = keras.preprocessing.image.load_img(
    './aw2.jpg', target_size=(180, 180)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print(
    "Resim {} gözüküyor. {:.2f} ihtimalle."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)