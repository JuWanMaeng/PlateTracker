## Results
![Tracking Result](result/result.gif)

## Installation
### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/JuWanMaeng/PlateTracker.git
cd PlateTracker
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```

## Acknowledgement
* This code is built on [ByteTrack](https://github.com/ifzhang/ByteTrack). We thank the authors for sharing their codes