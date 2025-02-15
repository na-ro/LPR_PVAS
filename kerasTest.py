import cv2
import keras_ocr
import matplotlib.pyplot as plt

def preprocess_image(image):
    # read image from specified path
    image = cv2.imread(image)
    # upscale the image by 4x using the linear interpolation technique
    resized_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

    # CLAHE to increase contrast (i have half a clue on how this works so far)
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    contrast_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # convert the image to grayscale for easier processing/OCR
    gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)

    # otsu thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # bitwise not to turn the white threshold text into black
    inverted_thresh = cv2.bitwise_not(thresh)

    # apply gaussian blur to sharpen image as much as possible
    # blur filter kernel specified at 9x9
    blurred = cv2.GaussianBlur(inverted_thresh, (9, 9), 0)

    # show fully processed image before returning and sending to KerasOCR
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(0)

    # convert image back to 3 channels from 1 (grayscale) as KerasOCR requires images to have 3
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)


# KerasOCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# process the image and perform OCR
image_path = 'sample.jpg'
processed_image = preprocess_image(image_path)
prediction_groups = pipeline.recognize([processed_image])

# display the image and plot the text results
fig, ax = plt.subplots(figsize=(10, 10))
keras_ocr.tools.drawAnnotations(image=processed_image, predictions=prediction_groups[0], ax=ax)
plt.show()
