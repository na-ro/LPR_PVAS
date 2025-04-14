# LPR Subsystem Explanation
## Branches
The **main branch** includes code and instructions to run this subsystem on the **Unigen Cupcake server** and uses security cameras connected through IP addresses.<br/>
The **Local-LiveFeed branch** includes code and instructions to run this subsystem **locally** (windows/Ubuntu computers that have integrated webcams). <br/>
Other branches were used for development.

## Dependency Explanation
**YOLO11:** Object detector. "YOLO systems enable real-time monitoring." Here is [Yolo's GitHub Repository](https://github.com/ultralytics/ultralytics) and the [documentation for YOLO](https://docs.ultralytics.com/). <br/>
**PaddleOCR:** Optional Character Recognition (OCR) - reads text from images to turn it into something useable by the subsystem. This team compared PaddleOCR with EasyOCR and found Paddle to be more accurate. The following are some helpful links: The [GitHub](https://github.com/PaddlePaddle/PaddleOCR) -- PaddleOCR is made by the company PaddlePaddle, which is a Chinese company. All code is available to be looked at in the GitHub, look through tools/README_en.md for the English version of the documentation --The [quickstart documentation](https://paddlepaddle.github.io/PaddleOCR/main/en/quick_start.html) and the [documentation home](https://paddlepaddle.github.io/PaddleOCR/main/en/index.html).<br/>
<br/>
The [README.md](README.md) contains instructions to install these dependencies.

## Subsystem's Current State
This subsystem can read license plates from either video or live feed. Video feed and image testing showed that the system is best able to read plates close to the camera. Plates that are further away and low camera quality in videos caused difficulty for the OCRs to read the plate number.<br/>
</br>
**Integrating on Unigen Cupcake Server:** When integrating this subsystem into the Cupcake Server, this team noticed that the Cupcake's processing speed was double the time it took to run locally on a laptop (500-600 ms to 1000-1200 ms). Because of low processing speeds, detections were changed to only be made every 10 frames, and car detection (which used to be implemented) was removed. The car detection used to reduce the amount of false positive mistakes the subsystem made since both a car and a license plate must be detected in the same frame. Since the detection process was the biggest factor in slowing the processing time, this functionality was mostly removed. </br> 
</br>
**Output:** Output is shown through the command line when this subsystem is run. Depending on if a license plate is detected, the system will attempt to run the OCR and then print whatever text the OCR reads. At the end of the current subsystem's code, there is a **temporary step for debugging** which **writes output to a csv file**. This is used only for testing the subsystem's accurary and ability to detect plates. In this CSV file, the frame number a license plate is detected is written, the text and it's confidence score, and the base64 conversion of the crop of the detected license plate. This step should eventually be replaced by calling the PCSO's computer LPR tool to check these results against their database of stolen vehicles and hotplates. It may also be possible to send the base64 image to the computer LPR tool so that it can check this subsystems accuracy and only check correct plates. Saving this crop if a hotplate/stolen vehicle is detected (or the screenshot at the time), could also be useful for the PCSO.

