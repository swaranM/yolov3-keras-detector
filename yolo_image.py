import cv2
import os
from yolov3 import *
from utils import *


def detect_image(cfg_path,nms_thresh,obj_threshold,image_path,labels_path,weights):
 
    net,blocks                = cfgparser(cfg_path)
    labels,no_classes,colors  = get_labels(labels_path,blocks)
    net_size                  = (int(net['height']),int(net['width']))
    nms_thresh                = nms_thresh
    anchors                   = [get_anchors(block) for block in blocks if 'yolo' in block['type']]
    if 'yolov3.h5' not in os.listdir("."):
        yolov3           = create_yolov3(blocks)
        print("Loading Weights....")
        yolov3           = load_weights(blocks,yolov3,weights)
        yolov3.save('yolov3.h5')
        print("Done.")
    else:
        yolov3           = load_model("yolov3.h5")    

    image                     = cv2.imread(image_path)
    new_image                 = get_letterbox_image(image,net_size)
    image_data                = np.array(new_image, dtype='float32')
    image_data                = image_data/255.
    image_data                = np.expand_dims(image_data,0)
    results                   = yolov3.predict(image_data)
    bboxes                    = [get_detections(i,j,net_size,no_classes,0.4) for i,j in zip(results,anchors)]
    bboxes                    = [i for i in bboxes if i]
    bboxes                    = [i for j in bboxes for i in j]            
    bboxes                    = [replace(bbox,label = labels[bbox.idx]) for bbox in bboxes]
    colors                    = {labels[k]:v for k,v in colors.items()}


    for _label in set([bbox.label for bbox in bboxes]):
        bbox_cls     = [bbox for bbox in bboxes if bbox.label == _label]
        bboxes_nms   = non_maximal_supression(bbox_cls,nms_thresh)
        for det in bboxes_nms:
            bbox = (get_scaled_boxes(det,labels,net_size,image))
            cv2.rectangle(image,(int(bbox.xmin),int(bbox.ymin)),(int(bbox.xmax),int(bbox.ymax)),colors[bbox.label], 2)
            cv2.rectangle(image,(int(bbox.xmin),int(bbox.ymin)),(int(bbox.xmin)+len(bbox.label)*12,int(bbox.ymin)+28),colors[bbox.label],-1)
            cv2.putText(image,bbox.label,(int(bbox.xmin),int(bbox.ymin)+12),cv2.FONT_HERSHEY_PLAIN,1, (0,0,0))
            cv2.putText(image,str(bbox.score)[:4],(int(bbox.xmin),int(bbox.ymin)+28),cv2.FONT_HERSHEY_PLAIN,1, (0,0,0))
    
    
    cv2.imshow("",image)
    cv2.imwrite(image_path.split(".")[0] + "_output.jpg",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    




