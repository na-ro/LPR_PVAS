import matplotlib.pyplot as plt

import keras_ocr

# initialize kerasOCR pipeline with pretrained models and weights
pipeline = keras_ocr.pipeline.Pipeline()

# array of image urls from various websites (literal copy-paste from url :D)
images = [
    keras_ocr.tools.read(url) for url in [
        'https://d.newsweek.com/en/full/1978635/florida-license-plate.webp?w=790&f=b4cc80cf8dc33f8cbcf7925146008e33',
        'https://preview.redd.it/what-license-plate-is-this-how-can-i-get-one-i-dont-see-it-v0-veh4vj3uhfbc1.jpeg?width=1080&crop=smart&auto=webp&s=93dec669038d540bfd5653475db04afc0cc0468b'
    ]
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

# plots the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

plt.show()