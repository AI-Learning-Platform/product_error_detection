# import the opencv library
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
# import the opencv library
import cv2
# define a video capture object
vid = cv2.VideoCapture(0)
img_width = 640
img_height = 480
vid.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
model = tf.keras.models.load_model('model/trained_model')
class_names = ['correct', 'wrong']

while(True):

    # Capture the video frame
    # by frame
    vid.set(cv2.CAP_PROP_EXPOSURE, -4)
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imwrite("caputer.jpg", frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    img = keras.preprocessing.image.load_img(
        'caputer.jpg', target_size=(img_width, img_height)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    print(
        "Resim {} gözüküyor. {:.2f} ihtimalle."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()