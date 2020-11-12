# USAGE
# python distraspmac.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel 

# import the necessary packages
#from imutils import build_montages
import datetime
import numpy as np
from imagezmq import imagezmq
import argparse
import imutils
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import time
from centroidtracker import CentroidTracker
from itertools import combinations
import math
import sys
import random
import os
from time import sleep




protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# load serialized model from disk
print("[INFO] loading model...")
tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)



def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))












fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
fps_start_time = datetime.datetime.now()
cap = cv2.VideoCapture(0)  



# start looping over all the frames
while True:
   
 
    now=datetime.datetime.now()

    ret,frame = cap.read()
    frame = imutils.resize(frame, width=320)
    total_frames = total_frames + 1
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
    detector.setInput(blob)
    person_detections = detector.forward()
    rects = []
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        if confidence > 0.6:
           idx = int(person_detections[0, 0, i, 1])
         

           if CLASSES[idx] != "person":
                    continue

           person_box = person_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
           (startX, startY, endX, endY) = person_box.astype("int")
           rects.append(person_box)

    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)
    centroid_dict = dict()
    objects = tracker.update(rects)
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)

        centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)


    for (id1, p1), (id2, p2), (id3,p3) in combinations(centroid_dict.items(), 3):
        dx1, dy1 = p1[0] - p2[0], p1[1] - p2[1]
        dx2, dy2= p1[0] - p3[0] , p1[1] - p3[1]
        dx3, dy3= p2[0] - p3[0] , p2[1] - p3[1]

        dis12 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        dis13= math.sqrt(dx2 * dx2 + dy2 * dy2)
        dis23= math.sqrt(dx3 * dx3 + dy3 * dy3)
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
       


        sys.stdout = open('raspmacdisfps.csv','a')
      

        print(str(now)+","+str(fps)+","+str(dis12)+","+str(dis13)+ ","+str(dis23)+"\n")
       


    cv2.imshow("Distance test", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()


