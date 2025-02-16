import cv2
from paddleocr import PaddleOCR
import numpy as np
import matplotlib.pyplot as plt

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

# declare/initialize PaddleOCR
# use_angle_cls enables angle classification, automatically correcting text orientation
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# process the image and perform OCR
image_path = 'help.jpg'
processed_image = preprocess_image(image_path)
result = ocr.ocr(processed_image, cls=True)

# display the results by looping through the returned results
# results are returned as an array of arrays with bounding box coordinates and text + confidence scores
for line in result[0]:
    text, prob = line[1][0], line[1][1]
    print(f"Detected text: {text} with probability {prob:.2f}")

# initialize the image canvas
fig, ax = plt.subplots(figsize=(15, 15))
ax.axis('off')  # hide axes

# draw the OCR boxes and texts using the returned results
for line in result[0]:
    # extract coordinates and covert them into a NumPy array for OpenCV
    (top_left, top_right, bottom_right, bottom_left) = line[0]
    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    # reshape points into the correct format for OpenCV 
    # (i was having problems with countless errors of formatting so this is from GPT; literally couldn't find an explanation anywhere else)
    points = points.reshape((-1, 1, 2))

    # draw the bounding box from the returned coordinates
    cv2.polylines(processed_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # convert the top_left coordinates to integers to use them for drawing the text
    top_left = tuple(map(int, top_left))

    # using the top_left coordinate, draw the detected text
    cv2.putText(processed_image, line[1][0], top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)

# show the license plate image with all the drawn results
ax.imshow(processed_image)
plt.show()
