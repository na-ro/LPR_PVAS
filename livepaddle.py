from ultralytics import YOLO
import cv2
from sort.sort import *
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import numpy as np

''' 
def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Also includes writing the base64 version of an image.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{}\n'.format('frame_nmr', 'text', 'text_score', 'cropped image'))

        for frame_nmr in results.keys():
            print(results[frame_nmr])
            if 'license_plate' in results[frame_nmr].keys() and 'text' in results[frame_nmr]['text'].keys():
                f.write('{},{},{},{}\n'.format(frame_nmr, results[frame_nmr]['text']['text'], results[frame_nmr]['text']['text_score']))
            else:
                f.write('ERROR: Couldnt find LP or text for util.py write_csv condition')
        f.close()
'''



#################
# Prepare PaddleOCR
#################
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
results = {} # results will be kept here. This is a temporary solution that should later be replaced with the 3rd party computer LPR system PSCO already has 

#################
# Load Live feed 
#################
cam = cv2.VideoCapture(0) 

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

'''
# Testing flat text detection with image
img_path = "./FL_plate_rsi407.webp"
result = ocr.ocr(img_path, cls=True)
if result[0] is not None:
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            text, prob = line[1][0], line[1][1]
            print(f"Detected text: {text} with probability {prob:.2f}")
'''

# start live feed
frame_nmr = -1

while True: # live
    frame_nmr += 1
    ret, frame = cam.read() 

    if ret: # live feed
        results[frame_nmr] = {}

        # Paddle check
        #img_path = (frame)[0]
        img_path = frame
        result = ocr.ocr(img_path, cls=True)
        if result[0] is not None:
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    text, prob = line[1][0], line[1][1]
                    print(f"Detected text: {text} with probability {prob:.2f}")


    # Show the captured 'frame' appears as live feed
    cv2.imshow('LPR live feed', frame)

    # Press 'q' to exit 
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture
cam.release()
cv2.destroyAllWindows()

# write results
# write_csv(results, './livepaddle.csv') 