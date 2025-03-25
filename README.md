# Licese Plate Reader for Patrol Vehicle Alert System (PVAS)

## References

Reference Video link: https://www.youtube.com/watch?v=fyJB1t0o0ms <br/>
Reference Github link: https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8 <br/>
Current timestamp: DONE <br/>

# Set-up (Ubuntu)

## Create (ana)conda environment 
```
 conda create --name NAME python=3.8
```
## Dependencies (--user might be necessary to run as admin)
```
pip install ultralytics --user
pip install scikit-image --user
pip install filterpy --user
pip install opencv-python --user
pip install paddleocr --user
python3 -m pip download paddlepaddle==2.4.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/noavx/stable.html --no-index --no-deps
python3 -m pip install NAME.whl
git clone https://github.com/abewley/sort.git
```

Checking if paddlepaddle is installed correctly:<br/>
Run <python3>, <import paddle> then <paddle.utils.run_check()> (It should return 'paddle installed successfully!' at the end of the outputs)

# Set-up (Windows)

## Create (ana)conda environment 
```
 Conda create --name ENV_NAME
```

## Dependencies (--user might be necessary to run as admin)
```
pip install ultralytics --user
pip install scikit-image --user
pip install filterpy --user
pip install opencv-python --user
```

## Clone SORT into root directory of project
Github link: https://github.com/abewley/sort <br/>

## Import Paddle
```
pip install paddlepaddle --user
git clone https://github.com/PaddlePaddle/PaddleOCR
Cd paddleOCR
pip3 install -r requirements.txt --user
Pip install paddleocr --user
```

# Requirements
Models should be in a models/ folder in the root directory <br/>
Sample video (Video with license plates to test code) should be in root directory called 'sample.mp4' <br/>
<br/>

## Download license plate model 
Direct link to model: https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view <br/>
Github link (will be in the Readme section): https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8 <br/>

## Sample video to test code
download link: https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/ <br/>
