import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# Florida-specific corrections
florida_char_corrections = {'B': '8', 'D': '0', 'Z': '2'}


def write_csv(results, output_path):
    """Write the results to a CSV file."""
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frameNumber', 'carID', 'boundingBox',
                                                'licensePlate', 'boundingBoxScore', 'text',
                                                'textScore'))

        for frameNumber in results.keys():
            for carID in results[frameNumber].keys():
                print(results[frameNumber][carID])
                if 'car' in results[frameNumber][carID].keys() and \
                        'licensePlate' in results[frameNumber][carID].keys() and \
                        'text' in results[frameNumber][carID]['licensePlate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frameNumber,
                                                            carID,
                                                            '[{} {} {} {}]'.format(
                                                                *results[frameNumber][carID]['car']['boundingBox']),
                                                            '[{} {} {} {}]'.format(
                                                                *results[frameNumber][carID]['licensePlate'][
                                                                    'boundingBox']),
                                                            results[frameNumber][carID]['licensePlate'][
                                                                'boundingBoxScore'],
                                                            results[frameNumber][carID]['licensePlate']['text'],
                                                            results[frameNumber][carID]['licensePlate']['textScore'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with Florida's format.

    Formats:
    - ABC1234
    - 123ABC
    - AB1 23C
    """
    text = text.replace(" ", "")  # Remove spaces for checking

    # Standard Florida format (ABC1234)
    if re.match(r"^[A-Z]{3}[0-9]{4}$", text):
        return True
    # Older Florida format (123ABC)
    if re.match(r"^[0-9]{3}[A-Z]{3}$", text):
        return True
    # Specialty plate format (AB1 23C)
    if re.match(r"^[A-Z]{2}[0-9]{1} [0-9]{2}[A-Z]{1}$", text):
        return True

    return False


def format_license(text):
    """Format the license plate text with character correction."""
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}

    for j in range(len(text)):
        char = text[j]

        # Apply existing character mappings
        if j in mapping and char in mapping[j]:
            char = mapping[j][char]

        # Apply Florida-specific corrections
        if char in florida_char_corrections:
            char = florida_char_corrections[char]

        license_plate_ += char

    return license_plate_


def read_license_plate(croppedLicensePlate):
    """Read the license plate text from the given cropped image."""
    detections = reader.readtext(croppedLicensePlate)

    for detection in detections:
        boundingBox, text, textScore = detection
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), textScore

    return None, None


def get_car(licensePlate, trackIDs):
    """Retrieve the vehicle coordinates and ID based on the license plate coordinates."""
    x1, y1, x2, y2, score, class_id = licensePlate

    foundLicensePlate = False
    for x in range(len(trackIDs)):
        xcar1, ycar1, xcar2, ycar2, carID = trackIDs[x]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            carIndex = x
            foundLicensePlate = True
            break

    if foundLicensePlate:
        return trackIDs[carIndex]

    return -1, -1, -1, -1, -1
