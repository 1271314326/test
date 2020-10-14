import cv2
import os
import time
import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
from PIL import Image, ImageDraw

print('[INFO] loading face detector model.......')
prototxt_path = r'./face_detector/deploy.prototxt'
weight_path = r'./face_detector/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxt_path, weight_path)

vs = VideoStream(0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 117.0, 120.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    locs = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            locs.append((startX, startY, endX, endY))

    for box in locs:
        mask_img = cv2.imread('./ma.jpg')
        (startX, startY, endX, endY) = box
        face_width = endX - startX
        face_height = endY - startY

        mask_img = np.asarray(mask_img)
        mask_img = Image.fromarray(mask_img)
        mask_img = mask_img.resize((int(face_width), int(face_height)))
        frame = Image.fromarray(frame)
        frame.paste(mask_img, (startX, startY))
        frame = np.asarray(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
