import cv2
import sys
import numpy as np
from keras.models import load_model
import json
from keras.models import model_from_json

IMG_SIZE = 100
def nothing(x):
    pass


def get_class_label(val, dictionary):
    for key, value in dictionary.items():
        if value == val:
            return key

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

# create alphabet dictionary to label the letters {'a':1, ..., 'nothing':29}

video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Model Image')

# set the ration of main video screen
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# set track bar of threshold values for Canny edge detection
# more on Canny edge detection here:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
cv2.createTrackbar('lower_threshold', 'Model Image', 0, 255, nothing)
cv2.createTrackbar('upper_threshold', 'Model Image', 0, 255, nothing)
cv2.setTrackbarPos('lower_threshold', 'Model Image', 100)
cv2.setTrackbarPos('upper_threshold', 'Model Image', 0)

while True:
    blank_image = np.zeros((100,800,3), np.uint8) # black image for the output
    ret, frame = video_capture.read() # capture frame-by-frame
    # set the corners for the square to initialize the model picture frame
    x_0 = int(frame.shape[1] * 0.1)
    y_0 = int(frame.shape[0] * 0.25)
    x_1 = int(x_0 + 200)
    y_1 = int(y_0 + 200)

    # MODEL IMAGE INITIALIZATION
    hand = frame.copy()[y_0:y_1, x_0:x_1] # crop model image
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY) # convert to grayscale
    # noise reduction
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    blured = cv2.erode(blured, None, iterations=2)
    blured = cv2.dilate(blured, None, iterations=2)
    #get the values from tack bar
    lower = cv2.getTrackbarPos('lower_threshold', 'Model Image')
    upper = cv2.getTrackbarPos('upper_threshold', 'Model Image')
    edged = cv2.Canny(blured,lower,upper) # aplly edge detector

    model_image = ~edged # invert colors
    model_image = cv2.resize( model_image,
        dsize=(IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC
    )
    model_image = np.array(model_image)
    model_image = model_image.astype('float32') / 255.0

    try:
        model_image = model_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        predict = model.predict(model_image)
        for values in predict:
            if np.all(values < 0.5):
                # if probability of each class is less than .5 return a message
                print ("Can't classify")
            else:
                predict = np.argmax(predict, axis=1)
                print (predict)
                
                
    except:
        pass


    
    # draw rectangle for hand placement
    cv2.rectangle(frame, (x_0, y_0), (x_1, y_1), (0, 255, 0), 2)

    # display the resulting frames
    cv2.imshow('Main Image', frame)
    cv2.imshow('Model Image', edged)
    
    key= cv2.waitKey(5)
    if key == 27:
          break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

