# Running PVAS Locally (machine with webcam)

## Set-up (Ubuntu)
The entire subsystem set-up should take approximately **9.3 GB** of space.

### Create (ana)conda environment 
Anaconda environments were used for easy dependency management per subystem. The following command creates an anaconda environment with python3.8 -- **Python3.8 is needed** for another requirement in the 'dependencies' section.
```
conda create --name NAME python=3.8
```
Other helpful conda commands: *$conda activate ENV_NAME* activates a conda enviornment, *$conda deactivate* will deactivate it. To list all conda enviornments on the machine, use *$conda env list*

### Dependencies
Most requirements can be installed using pip (--user might be necessary to run as admin):
```
pip install ultralytics --user
pip install scikit-image --user
pip install filterpy --user
pip install opencv-python --user
pip install paddleocr --user
git clone https://github.com/abewley/sort.git
```
Installing paddlepaddle requires extra commands on the Ubuntu Unigen Cupcake machine, since the machine doesn't support avx. Instead we'll install the Paddle package of noavx (note: the noavx paddle version only supports python3.8, which is why the conda environment is created with python3.8). We installed the cpu and mkl version for a noavx machine. <br/>
The latest version of paddlepaddle -- as of 3/25/25 -- that supports noavx installations is 2.4.2. Find the entire install instructions [here](https://www.paddlepaddle.org.cn/documentation/docs/en/2.4/install/pip/linux-pip_en.html)
```
python3 -m pip download paddlepaddle==2.4.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/noavx/stable.html --no-index --no-deps
python3 -m pip install NAME.whl
```
To test if paddlepaddle has been successfully installed, run *$python3*, *$import paddle* then *$paddle.utils.run_check()* (It should return 'paddle installed successfully!' at the end of the outputs)

## Set-up (Windows)

### Create (ana)conda environment 
In windows, paddlepaddle can be installed normally with pip. This means any python version can be used, unlike in Ubuntu machines.
```
 conda create --name ENV_NAME
```

### Dependencies (--user might be necessary to run as admin)
Install all dependencies using pip (--user might be necessary to run as admin):
```
pip install ultralytics --user
pip install scikit-image --user
pip install filterpy --user
pip install opencv-python --user
pip install paddleocr --user
pip install paddlepaddle --user
git clone https://github.com/abewley/sort
```

## Requirements
Models should be in a models/ folder in the root directory <br/>
