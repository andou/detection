FROM python:3.8

RUN apt-get update

RUN apt-get install tesseract-ocr ffmpeg libsm6 libxext6  -y


RUN pip install --upgrade pip

RUN pip install pytesseract
RUN pip install opencv-python-headless
RUN pip install numpy

WORKDIR /opt/detect/

COPY run.py /opt/detect/
COPY classifiers /opt/detect/classifiers/

CMD python run.py