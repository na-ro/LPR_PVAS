import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
    frameNumbers = np.array([int(row['frameNumber']) for row in data])
    carIDs = np.array([int(float(row['carID'])) for row in data])
    boundingBoxes = np.array([list(map(float, row['boundingBox'][1:-1].split())) for row in data])
    licensePlates = np.array([list(map(float, row['licensePlate'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_carIDs = np.unique(carIDs)
    for carID in unique_carIDs:
        frameNumbers_ = [p['frameNumber'] for p in data if int(float(p['carID'])) == int(float(carID))]
        print(frameNumbers_, carID)

        # Filter data for a specific car ID
        car_mask = carIDs == carID
        car_frameNumbers = frameNumbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_interpolated = []

        first_frameNumber = car_frameNumbers[0]
        last_frameNumber = car_frameNumbers[-1]

        for i in range(len(boundingBoxes[car_mask])):
            frameNumber = car_frameNumbers[i]
            boundingBox = boundingBoxes[car_mask][i]
            licensePlate = licensePlates[car_mask][i]

            if i > 0:
                prev_frameNumber = car_frameNumbers[i - 1]
                prev_boundingBox = car_bboxes_interpolated[-1]
                prev_licensePlate = license_plate_interpolated[-1]

                if frameNumber - prev_frameNumber > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frameNumber - prev_frameNumber
                    x = np.array([prev_frameNumber, frameNumber])
                    x_new = np.linspace(prev_frameNumber, frameNumber, num=frames_gap, endpoint=False)

                    interp_func = interp1d(x, np.vstack((prev_boundingBox, boundingBox)), axis=0, kind='linear')
                    interpolated_boundingBoxes = interp_func(x_new)

                    interp_func = interp1d(x, np.vstack((prev_licensePlate, licensePlate)), axis=0, kind='linear')
                    interpolated_licensePlates = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_boundingBoxes[1:])
                    license_plate_interpolated.extend(interpolated_licensePlates[1:])

            car_bboxes_interpolated.append(boundingBox)
            license_plate_interpolated.append(licensePlate)

        for i in range(len(car_bboxes_interpolated)):
            frameNumber = first_frameNumber + i
            row = {}
            row['frameNumber'] = str(frameNumber)
            row['carID'] = str(carID)
            row['boundingBox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['licensePlate'] = ' '.join(map(str, license_plate_interpolated[i]))

            if str(frameNumber) not in frameNumbers_:
                # Imputed row, set the following fields to '0'
                row['boundingBoxScore'] = '0'
                row['text'] = '0'
                row['textScore'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frameNumber']) == frameNumber and int(float(p['carID'])) == int(float(carID))][0]
                row['boundingBoxScore'] = original_row['boundingBoxScore'] if 'boundingBoxScore' in original_row else '0'
                row['text'] = original_row['text'] if 'text' in original_row else '0'
                row['textScore'] = original_row['textScore'] if 'textScore' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

# Load the CSV file
with open('lpr.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frameNumber', 'carID', 'boundingBox', 'licensePlate', 'boundingBoxScore', 'text', 'textScore']
with open('interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
