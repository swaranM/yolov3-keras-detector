import argparse
from yolo_image import detect_image
from yolo_video import detect_video


parser = argparse.ArgumentParser(description='YoloV3 Detector')
parser.add_argument("-detect_image", "--image_path", type = str, help = "Path of the image")
parser.add_argument("-detect_video", "--video_path", type = str, help = "Path of the video")
parser.add_argument("-labels"      , "--labels_path",type = str, help = "Path of labels",default="labels.txt")
parser.add_argument("-nms"         , "--nms_thresh",type = float, help = "Non Maximal Supression", default=0.45)
parser.add_argument("-thresh"      , "--obj_thresh",type = float, help = "Object Threshold"   , default=0.5)
parser.add_argument("-cfg_path"    , "--cfg",type= str,help="Path of Configuration file" , default="cfg/yolov3.cfg")
parser.add_argument("-weights_path", "--weights",type= str,help="Path of weights file"   , default="yolov3.weights")

args = parser.parse_args()

if args.image_path:
    detect_image(args.cfg,args.nms_thresh,args.obj_thresh,args.image_path,args.labels_path,args.weights)

if args.video_path:
    detect_video(args.cfg,args.nms_thresh,args.obj_thresh,args.video_path,args.labels_path,args.weights)
