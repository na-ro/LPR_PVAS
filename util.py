# for image base64 encoding
import base64
import cv2
from paddleocr import PaddleOCR, draw_ocr

# Initialize the OCR reader
# Paddleocr supports Chinese, English, French, German, Korean and Japanese
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan` to switch the language model in order
#################
# Load PaddleOCR
#################
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

def preprocess_image(image):
    # read image from specified path
    image = cv2.imread(image)
    # upscale the image by 4x with the LANCZOS4 interpolation technique (makes edges sharper and upscales the most)
    resized_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

    # apply gaussian blur to sharpen image as much as possible
    # filter kernel size specified at 7x7
    blurred = cv2.GaussianBlur(resized_image, (7,7), 0)

    # denoise the image
    # parameters: h=9 - noise removal strength of 9, templateWindowSize=10 - size of window to compare neighboring pixels,
    # searchWindowSize=21 - size of window to search for similar patches and average them
    denoised = cv2.fastNlMeansDenoisingColored(blurred, None, h=9, templateWindowSize=10, searchWindowSize=21)

    # apply CLAHE contrast increase
    # convert to LAB space (3 channels: Luminance, A = Green-Red, B = Blue-Yellow)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # actually create the CLAHE contrast filter to apply to the image
    # clipLimit=7 specifies the contrast enhancement, with higher values being stronger
    # tileGridSize=(25, 25) splits the image into the specified grid size to apply the enhancement
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(25, 25))

    # apply the contrast enhancement only in the Luminance channel
    l = clahe.apply(l)

    # merge the channels back together and convert back into BGR space for PaddleOCR
    lab = cv2.merge((l, a, b))
    contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # convert the image to grayscale for easier processing/OCR
    gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)

    # otsu thresholding
    # thresh=0 is automatically calculated
    # maxval=255 specifies the max intensity value for pixels going through the threshold. in this case, 255 is white
    # if a pixel ends up with a value greater than the calculated thresh, that pixel is then set to black, being set to white if otherwise.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # invert the thresholded image to turn all white text into black
    # PaddleOCR seems to perform better with BOW instead of WOB in most cases
    inverted_thresh = cv2.bitwise_not(thresh)

    cv2.imshow('img', inverted_thresh)
    cv2.waitKey(0)

    # convert grayscale/thresh image back to 3 channels, as PaddleOCR requires 3
    return cv2.cvtColor(inverted_thresh, cv2.COLOR_GRAY2RGB)

def image_to_base64(image):
    """
    Convert an OpenCV image (NumPy array) to a Base64 string.
    
    :param image: OpenCV image (NumPy array).
    :return: Base64 encoded string.
    """
    # Encode the image as JPEG (can also use PNG)
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    # Convert to Base64
    return base64.b64encode(image_bytes).decode('utf-8')


def write_csv_wimage(results, output_path):
    """
    Write the results to a CSV file.

    Also includes writing the base64 version of an image.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'license_number', 'license_number_score', 'image'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                base64_image = image_to_base64(results[frame_nmr][car_id]['license_plate']['image'])
                print(results[frame_nmr][car_id])
                if 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{}\n'.format(frame_nmr, car_id, results[frame_nmr][car_id]['license_plate']['text'], results[frame_nmr][car_id]['license_plate']['text_score'], base64_image))
                else:
                    f.write('ERROR: Couldnt find LP or text for util.py write_csv_X condition')
        f.close()

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Also includes writing the base64 version of an image.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'license_number', 'license_number_score', 'cropped image', 'processed image'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                croppedImage_b64 = image_to_base64(results[frame_nmr][car_id]['license_plate']['cropped image'])
                processedImage_b64 = image_to_base64(results[frame_nmr][car_id]['license_plate']['processed image'])
                print(results[frame_nmr][car_id])
                if 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{}\n'.format(frame_nmr, car_id, results[frame_nmr][car_id]['license_plate']['text'], results[frame_nmr][car_id]['license_plate']['text_score'], croppedImage_b64, processedImage_b64))
                else:
                    f.write('ERROR: Couldnt find LP or text for util.py write_csv condition')
        f.close()


def license_complies_format(text):
    #if len(text) != 7:
    if len(text) < 6 or len(text) > 8:
        return False

    return True
    
''' 
def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    # depending on what is expected (2 letters, 2 numbers, 3 letters), the characters will be switched to match format
    license_plate_ = ''
    #mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
    #           2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_
'''

def paddle_read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.
    """
    # process the image and perform OCR
    processed_image = preprocess_image(license_plate_crop)
    #paddleOCRresult = ocr.ocr(processed_image, cls=True)

    paddleOCRresult = ocr.ocr(processed_image, det=False, cls=False)
    
    for idx in range(len(paddleOCRresult)):
        res = paddleOCRresult[idx]
        for line in res:
            text, score = line
            text = text.upper().replace(' ', '')
    
        return text, score
    
    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates

    Returns:
        tuple: Tuple contianing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """     
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    
    if foundIt:
        return vehicle_track_ids[car_indx]
    
    return -1, -1, -1, -1, -1