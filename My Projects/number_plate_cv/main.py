import cv2 as cv
import easyocr
import imutils
import numpy as np
import supres

# min width and height of the square of the object
min_width_react = 100
min_height_react = 100

# the cascade for the classification of the autos
auto_classifier = cv.CascadeClassifier('cvcars/cars.xml')

# camera capture
camera = cv.VideoCapture('cvcars/video.mp4')

# recognition of letters and numbers
reader = easyocr.Reader(['en'])
# the model for enlarging image pixels
generator = supres.load_model('gen_e_1.h5', compile=False)

while(True):
    # get image
    ret, img = camera.read()
    # color the image in gray tones
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # to reduce the image noise
    blur = cv.GaussianBlur(gray, (3,3), 5)
    # car detection in the image
    autos = auto_classifier.detectMultiScale(blur)
    
    # for every coordinates of each car in the image
    for (x,y,w,h) in autos:
        # validation of these coordinates
        validate_counter = (w >= min_width_react) and (h >= min_height_react) 
        if not validate_counter:
            continue
        # cut a photo of a car in the image
        auto_img = img[y:y+h, x:x+w]
        # color the image in gray tones
        gray = cv.cvtColor(auto_img, cv.COLOR_BGR2GRAY)

        # Noise reduction
        bfilter = cv.bilateralFilter(gray, 11, 11, 17)
        # Edge detection
        edged = cv.Canny(bfilter, 30, 200)

        # find countours of each car in the image
        keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # get these contours
        contours = imutils.grab_contours(keypoints)
        # sort these contours
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]
        
        location = None
        # detect just in the image of a car
        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.018 * perimeter, True)
            # number of edges
            if len(approx) == 4: 
                    location = approx
        # if number plare is detected in the image
        if location is not None:
            # make mask of the number plate
            mask = np.zeros(auto_img.shape[:2], np.uint8)
            new_image = cv.drawContours(mask, [location], 0, 255, -1)
            # crop the image of a the plate number
            (x_ph, y_ph) = np.where(mask==255)
            (x1, y1) = (np.min(x_ph), np.min(y_ph))
            (x2, y2) = (np.max(x_ph), np.max(y_ph))
            cropped_image = auto_img[x1:x2+1, y1:y2+1]
            # improve image equality
            image = cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB)
            image = image / 255.
            image = np.expand_dims(image, axis=0)
            imp_image = generator.predict(image)
            # recognize the number plate
            num_plate = reader.readtext(imp_image)
            # if reader recognized the license plate
            if num_plate != []:
                # save the image of the license plate
                cv.imwrite(f'photo_num{num_plate[0]}.png', imp_image)
                # save the image of the auto
                cv.imwrite(f'photo_car{num_plate[0]}.png', auto_img)
                # write text of the license plate in the window
                cv.putText(img, f'{num_plate}', (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # show the car in the window
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        # write text in the window
        cv.putText(img, 'Car', (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # show the window
    cv.imshow('LIVE', img)
    key = cv.waitKey(1)

    if key==27:
        break

cv.destroyAllWindows()
camera.release()