import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# define a video capture object
from pandas import np

vid = cv2.VideoCapture(0)
cameraName = vid.getBackendName()

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(cameraName)
img_counter = 1

while(True):

    # Capture the video frame
    # by frame

    vid.set(cv2.CAP_PROP_EXPOSURE, -4)
    ret, frame = vid.read()
    img_name = "dataset/raw/dataset_{}.jpg".format(img_counter)
    img_counter += 1
    # Display the resulting frame
    if (img_counter % 5) == 0:
        #gray = tf.image.rgb_to_grayscale(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_name, gray)
        cv2.imshow('frame', gray)

# the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

