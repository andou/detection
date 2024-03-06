import cv2
from pytesseract import pytesseract
from pytesseract import Output
import numpy as np
import os

images_sample_folder = '/var/images/sample/'
images_redact_folder = '/var/images/redacted/'

for image_file in os.listdir(images_sample_folder):
    image = cv2.imread(images_sample_folder + image_file)

    # Create a CascadeClassifier object for face detection
    face_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_frontalface_default.xml')

    # Create a CascadeClassifier object for license plate detection
    # plate_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_russian_plate_number.xml')
    # plate_cascade = cv2.CascadeClassifier('./classifiers/haarcascade_license_plate_rus_16stages.xml')
    plate_cascade = cv2.CascadeClassifier('./classifiers/GreenParking_num-3000-LBP_mode-ALL_w-30_h-20.xml')

    text_tesseract = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(text_tesseract['level'])


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces and license plates in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Optional: Draw rectangles around the detected faces and license plates
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 10)

    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 10)

    for i in range(n_boxes):
        (x, y, w, h) = (text_tesseract['left'][i], text_tesseract['top'][i], text_tesseract['width'][i], text_tesseract['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # Apply Gaussian blurring to the detected faces and license plates
    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (255, 255), 0)
        image[y:y+h, x:x+w] = blurred_roi

    for (x, y, w, h) in plates:
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (255, 255), 0)
        image[y:y+h, x:x+w] = blurred_roi


    # Save the anonymized image to disk
    cv2.imwrite(images_redact_folder + image_file, image)