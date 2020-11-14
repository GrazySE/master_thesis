# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from centroidtracker import CentroidTracker
from itertools import combinations
import math
import sys
import random
import datetime
import os
from time import sleep


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-u", "--movidius", type=bool, default=0,
    help="boolean indicating if the Movidius should be used")
args = vars(ap.parse_args())


# construct the argument parse and parse the arguments

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]



tracker = CentroidTracker(maxDisappeared=40, maxDistance=5)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
detector = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# specify the target device as the Myriad processor on the NCS
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()
# loop over the frames from the video stream


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
#Pass the video link here












fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
fps_start_time = datetime.datetime.now()
 



# start looping over all the frames
while True:
   
 
    now=datetime.datetime.now()

    frame = vs.read()
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

        #text = "ID: {}".format(objectId)
        #cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

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
            
          
        sys.stdout = open('raspdisfps.csv','a')
        print(str(now)+","+str(fps)+","+str(dis12)+","+str(dis13)+ ","+str(dis23)+"\n")
#         cv2.putText(frame, str(fps),  (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        

   
        




#         distance_text1 = "Distance A - B: {:.2f}".format(dis12)
#         distance_text2 = "Distance A - C: {:.2f}".format(dis13)
#         distance_text3 = "Distance B - C: {:.2f}".format(dis23)
          
       


#         
#         cv2.putText(frame, distance_text2,  (5, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
#         cv2.putText(frame, distance_text3,  (5, 170), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

      




#         if distance < 75.0:
#                 if id1 not in red_zone_list:
#                     red_zone_list.append(id1)
#                 if id2 not in red_zone_list:
#                     red_zone_list.append(id2)               
     
         
                
    



      


#     for id, box in centroid_dict.items():
    
    cv2.imshow("Distance test", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()


