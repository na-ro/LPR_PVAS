import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # Top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # Bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # Top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # Bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


results = pd.read_csv('./interpolated.csv')

# Load video
video_path = './sample.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

licensePlateData = {}
for carID in np.unique(results['carID']):
    maxScore = np.amax(results[results['carID'] == carID]['textScore'])
    licensePlateData[carID] = {
        'licenseCrop': None,
        'licensePlateNumber': results[(results['carID'] == carID) & (results['textScore'] == maxScore)]['text'].iloc[0]
    }
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['carID'] == carID) & (results['textScore'] == maxScore)]['frameNumber'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['carID'] == carID) & (results['textScore'] == maxScore)]['licensePlate'].iloc[0]
                                      .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    licenseCrop = frame[int(y1):int(y2), int(x1):int(x2), :]
    licenseCrop = cv2.resize(licenseCrop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    licensePlateData[carID]['licenseCrop'] = licenseCrop

frameNumber = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames
ret = True
while ret:
    ret, frame = cap.read()
    frameNumber += 1
    if ret:
        df_ = results[results['frameNumber'] == frameNumber]
        for rowIndex in range(len(df_)):
            # Draw car bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[rowIndex]['boundingBox']
                                                              .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 5,
                        line_length_x=200, line_length_y=200)

            # Draw license plate bounding box
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[rowIndex]['licensePlate']
                                              .replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Place cropped license plate image
            licenseCrop = licensePlateData[df_.iloc[rowIndex]['carID']]['licenseCrop']
            H, W, _ = licenseCrop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = licenseCrop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (textWidth, textHeight), _ = cv2.getTextSize(
                    licensePlateData[df_.iloc[rowIndex]['carID']]['licensePlateNumber'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            licensePlateData[df_.iloc[rowIndex]['carID']]['licensePlateNumber'],
                            (int((car_x2 + car_x1 - textWidth) / 2), int(car_y1 - H - 250 + (textHeight / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except:
                pass

        out.write(frame)
        #frame = cv2.resize(frame, (1280, 720))

out.release()
cap.release()