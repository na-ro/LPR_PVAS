# Will be able to read UK license plates. Modify pipeline to apply to other lp's

from ultralytics import YOLO
import cv2

from sort.sort import *
from util import get_car, read_license_plate, write_csv_wimage

results = {}
mot_tracker = Sort() # Object that can sort. Object trackers to track all vehicles

# Load two models. One detects cars, the other detects license plates
# Model trained on coco dataset. Pretrained model from ultralytics. Used to detect cars.
#################
# load models 
#################
coco_model = YOLO('./models/yolo11n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt') 

#################
# load video
#################
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7] # Detect [car, motorbike, bus, truck]

#################
# read frames
#################
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    #if ret and frame_nmr < 20: # uncomment this and comment the next line for short testing
    if ret:
        results[frame_nmr] = {}
        #detect vehicles
        detections = coco_model(frame)[0]
        detections_ = [] # bounding boxes
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles (object tracking)
        track_ids = mot_tracker.update(np.asanyarray(detections_))
        
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plates
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plates
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read the license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None and license_plate_text_score >= 0.7:
                    results[frame_nmr][car_id] = {'license_plate': {'text': license_plate_text, 'text_score': license_plate_text_score, 'image': license_plate_crop}}

# write results
write_csv_wimage(results, './tempTEST.csv')