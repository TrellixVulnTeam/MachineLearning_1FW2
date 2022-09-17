# import the necessary packages
from keras.utils.image_utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2 as cv

# constants
SCREEN_HEIGHT = 720
SCREEN_WIDTH = 1280
FACE_W = 600
FACE_H = 600

# find the starting points for the facebox
def face_coords(SCREEN_HEIGHT, SCREEN_WIDTH, FACE_W, FACE_H):
    # find the center of the screen
    x, y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
    # divide the height and width of the facebox to place them in the center
    w, h = FACE_W / 2, FACE_H / 2
    # starting points for the facebox
    x, y = x - w, y - h
    return int(x), int(y)
    
# find the coordinates for a facebox
face_x, face_y  = face_coords(SCREEN_HEIGHT, SCREEN_WIDTH, FACE_W, FACE_H)

# four points of the facebox 
LEFT_UP_FACEBOX = (face_x, face_y)
LEFT_DOWN_FACEBOX = (face_x, face_y+FACE_H)
RIGHT_DOWN_FACEBOX = (face_x+FACE_W, face_y+FACE_H)
RIGHT_UP_FACEBOX = (face_x+FACE_W, face_y)

# load the face detector cascade and smile detector CNN
face_detector = cv.CascadeClassifier('/Cascades/haarcascade_frontalface_default.xml')
eye_detector = cv.CascadeClassifier('/Cascades/haarcascade_eye.xml')
model = load_model('/Models/model.h5')

# load video or photo
camera = cv.VideoCapture('/photos/test1.jpg')

# keep looping
while True:
    # grab the current frame
    _, frame = camera.read()

    # resize the fram, convert it to grayscale, and then clone the
    # orgignal frame so we draw on it later in the program
    frame = imutils.resize(frame, width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    frameClone = cv.resize(frameClone, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # detect faces in the input frame, then clone the frame so that we can draw onit
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    # detect eyes in the photo
    eyes = eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    # amount of the eyes
    num_eyes = len(eyes)
    # display the facebox
    cv.rectangle(frameClone, (face_x, face_y), (face_x+FACE_W, face_y+FACE_H), (0,255,0), 1)
    # for every face
    for (x, y, w, h) in faces:
        # four points of the face frame
        LEFT_UP_FACE = (x, y) # leftUp
        LEFT_DOWN_FACE = (x, y+h) # rightDown
        RIGHT_DOWN_FACE = (x+w, y+h) # leftDown
        RIGHT_UP_FACE = (x+w, y) # rightUp
        
        # extract the ROI of the face from the grayscale image
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        cv.rectangle(frameClone, (x, y), (x+w, y+h), (0,0,255), 1)
        roi = gray[y:y + h, x:x + w]
        roi = cv.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probaboilities of both 'smiling' and 'not smiling',
        # then set the label accordingly
        (notSmiling, Smiling) = model.predict(roi, verbose=0)[0]
        label = 'Smiling' if Smiling > notSmiling else "Not Smiling"

        # determine the position of the face to the facebox
        # upper and right frame line -> left frame line -> bottom line of the frame
        if RIGHT_UP_FACE[0] < RIGHT_UP_FACEBOX[0] and RIGHT_UP_FACE[1] > RIGHT_UP_FACEBOX[1] and \
           LEFT_UP_FACE[0] > LEFT_UP_FACEBOX[0] and LEFT_UP_FACE[1] > LEFT_UP_FACEBOX[1] and \
           RIGHT_DOWN_FACE[1] < RIGHT_DOWN_FACEBOX[1]:
            # display the face frame
            cv.rectangle(frameClone, (x, y), (x+w, y+h), (0,255,0), 1)
            # displayed information for the user
            info = 'Please leave your head in this position'
            if label == 'Smiling':
                info = "Please don't smile"
            else:
                # if user's eyes aren't visible
                if num_eyes < 2:
                    info = 'Please remove the hairstyle from your eyes'
                else:
                    # save the photo
                    cv.imwrite('photo.jpg', frame[face_y:face_y + FACE_H, face_x:face_x + FACE_W])
                    # end program execution
                    exit()
        else:
            # user's face isn't in the facebox
            info = 'Please fix your face inside the square on the display'
        # display the information for user
        cv.putText(frameClone, info, (50, 35), cv.QT_FONT_NORMAL, 0.95, (0, 0, 0), 3)

    # show user's window
    cv.imshow('Face', frameClone)

    # if 'q' key is pressed, stop the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv.destroyAllWindows()
