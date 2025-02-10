import cap
from ultralytics import YOLO

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import cv2

results = {}
objTracker = Sort()

# load models to use for license plate detection/general preprocessing
cocoModel = YOLO('yolo11n.pt') # model used to detect cars, specifically
detectorLPR = YOLO('models/LPRModel.pt') # model used to detect license plates


# load video feed
vid = cv2.VideoCapture('./sample.mp4')
vehicles = [2, 3, 5, 7] # array containing id's of vehicle types (2 = car, 3 = bus, etc.)

# read frames from video
frameNumber = -1
read = True
while read:
        frameNumber += 1
        read, frame = vid.read()
        if read:
                results[frameNumber] = {}
                #detect vehicles from frame
                vehicleDetections = cocoModel(frame)[0]
                vehicleBoundingBoxes = []
                for detection in vehicleDetections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection # at every point, we are detecting the bounding box's coordinates, confidence score, and class id (what object it classifies as)
                        if int(class_id) in vehicles: #if it's a vehicle, append it to the bounding boxes array
                                vehicleBoundingBoxes.append([x1, y1, x2, y2, score])
                # track vehicles moving through video
                trackIDs = objTracker.update(np.asarray(vehicleBoundingBoxes)) #assigning an ID to each car we have detected using the bounding boxes

                # detect license plates
                # this is done using the same method used above to detect the vehicles from the frame (lines 25-28)
                licensePlates = detectorLPR(frame)[0]
                for licensePlate in licensePlates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = licensePlate

                        # assign each license plate to a car
                        xcar1, ycar1, xcar2, ycar2, carID = get_car(licensePlate, trackIDs) #will return the coordinates of the car the license plate belongs to

                        if carID != -1:

                                # crop each specific license plate
                                croppedLicensePlate = frame[int(y1):int(y2), int(x1): int(x2), :]

                                # process license plate image to make it easier to read
                                grayLicensePlate = cv2.cvtColor(croppedLicensePlate, cv2.COLOR_BGR2GRAY)
                                # _, licensePlateThreshold = cv2.threshold(grayLicensePlate, 64, 255, cv2.THRESH_BINARY_INV) don't need this since the grayscale image performs better on average

                                # read each license plate number
                                licensePlateNumberText, licensePlateTextConfidenceScore = read_license_plate(croppedLicensePlate)

                                if licensePlateNumberText is not None:
                                        results[frameNumber][carID] = {'car': {'boundingBox': [xcar1, ycar1, xcar2, ycar2]},
                                                                       'licensePlate': {'boundingBox': [x1, y1, x2, y2],
                                                                                        'text': licensePlateNumberText,
                                                                                        'boundingBoxScore': score,
                                                                                        'textScore': licensePlateTextConfidenceScore}}
# write the results
write_csv(results, './lpr.csv')