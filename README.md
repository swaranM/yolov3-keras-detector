# Yolov3 Object Detector
Yolov3 Detector written in keras.



## Usage

```
Detecting objects on image:
python main.py -detect_image image_path


Detecting objects on video:
python main.py -detect_video video_path



Optional Arguments:
-nms (default = 0.45),-thresh(default = 0.5),-labels(default = labels.txt),
-cfg_path(default = "cfg/yolov3.cfg"), -weights_path(default = "yolov3.weights")


** Note:
yolov3.weights have to be downloaded. It can be found on pjreddie.com/yolo
```

## Example

```
python main.py -detect_image "samples/harden.jpg"
```
![harden](https://github.com/swaran24697/yolov3-keras-detector/blob/master/samples/harden_output.jpg)


## Requirements
```
python     =>3.7
tensorflow =>1.15
keras      = 2.3.1
OpenCV     =>3.2.0
numpy
```
