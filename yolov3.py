import tensorflow as tf
import keras
import cv2
import numpy as np
import time
from utils import *
from keras.layers import Conv2D 
from keras.layers import Input, BatchNormalization, LeakyReLU
from keras.layers import ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model,load_model
from keras.utils.conv_utils import convert_kernel
from dataclasses import dataclass,astuple,replace



def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_anchors(block):

    anchors = []
    masks = block['mask'].split(",")
    anchor = block['anchors'].split()
    for i in masks:
        anchors.append(anchor[int(i)])
    return [i.strip(",") for i in anchors]


def get_detections(result,anchors,dim,no_classes,threshold):

    grid_h, grid_w  = result.shape[1:3]
    stride          = dim[0] // result.shape[1]
    grid            = dim[0] // stride
    result          = result.reshape(1,grid*grid,result.shape[-1])
    result          = result.reshape(1,grid*grid*3,no_classes + 5)

    result[:,:,:2]  = sigmoid(result[:,:,:2])
    result[:,:,4:]  = sigmoid(result[:,:,4:])

    anchors         = [tuple(i.split(",")) for i in anchors]
    anchors         = [(int(i[0])/stride,int(i[1])/stride) for i in anchors]
    anchors         = np.tile(anchors,(grid*grid,1))


    offsets         = np.array([[j,i] for i in range(grid_h) for j in range(grid_w)])
    offsets         = np.repeat(offsets,3,0)
    result[:,:,0]   = (result[:,:,0] + offsets[:,0]) 
    result[:,:,1]   = (result[:,:,1] + offsets[:,1]) 
    
    result[:,:,2]   = np.exp(result[:,:,2]) * anchors[:,0] 
    result[:,:,3]   = np.exp(result[:,:,3]) * anchors[:,1] 


    indices         = result[:,:,4] > threshold
    result          = result[indices]
    dummy           = np.copy(result)

    result[:,0]     = dummy[:,0] - dummy[:,2]/2
    result[:,1]     = dummy[:,1] - dummy[:,3]/2
    result[:,2]     = dummy[:,0] + dummy[:,2]/2
    result[:,3]     = dummy[:,1] + dummy[:,3]/2
    result[:,:4]    = result[:,:4] * stride  
    
    result          = result[np.argsort(-result[:,4])]
    max_class       = np.amax(result[:,5:no_classes], axis=1)  
    max_class_idx   = np.argmax(result[:,5:no_classes], axis=1)
    bounding_boxes  = [BBox(res[0],res[1],res[2],res[3],res[4],score,idx) for res,score,idx in zip(result, max_class, max_class_idx)]
    
    if bounding_boxes:
        return bounding_boxes
  

def non_maximal_supression(bboxes,nms_thresh):

    ret = []  
    while len(bboxes) > 0:
        bb1                         = bboxes[0]
        ret.append(bboxes[0])
        bb1_x1,bb1_y1,bb1_x2,bb1_y2 = astuple(bb1)[0:4]
        bb1_area                    = (bb1_x2 - bb1_x1 + 1) * (bb1_y2 - bb1_y1 + 1)
        bboxes                      = bboxes[1:]
        if bboxes:
            bb2         = np.array([astuple(i)[0:4] for i in bboxes])
            bb2_x1      = bb2[:,0]
            bb2_y1      = bb2[:,1]
            bb2_x2      = bb2[:,2]
            bb2_y2      = bb2[:,3]
            
            
            bb2_areas   = (bb2_x2 - bb2_x1 + 1) *  (bb2_y2 - bb2_y1 + 1)
            
            xx1         = np.maximum(bb1_x1,bb2_x1)
            yy1         = np.maximum(bb1_y1,bb2_y1)
            xx2         = np.minimum(bb1_x2,bb2_x2)
            yy2         = np.minimum(bb1_y2,bb2_y2)
            inter       = (xx2 - xx1 + 1)       *  (yy2 - yy1 + 1)

            iou         = inter / (bb1_area + bb2_areas - inter)
            indices     = [i for i,j in enumerate(iou)  if j < nms_thresh]
            bboxes      = [v for i,v in enumerate(bboxes) if i in indices]
        
    return ret

def get_scaled_boxes(bbox,label_dict,net_size,original_image):
    
    img_h,img_w          = net_size
    target_h,target_w,_  = original_image.shape
    r                    = min(img_w/target_w,img_h/target_h)

    bbox                 = replace(bbox,xmin = (bbox.xmin - ((img_w - target_w * r) / 2)) * 1/r)
    bbox                 = replace(bbox,xmax = (bbox.xmax - ((img_w - target_w * r) / 2)) * 1/r)
    bbox                 = replace(bbox,ymin = (bbox.ymin - ((img_h - target_h * r) / 2)) * 1/r)
    bbox                 = replace(bbox,ymax = (bbox.ymax - ((img_h - target_h * r) / 2)) * 1/r)
    
    return bbox

@dataclass
class BBox:
    xmin     : float
    ymin     : float
    xmax     : float
    ymax     : float
    objness  : float
    score    : float
    idx      : int
    label    : str = None

class Conv2DLayer:

    def __init__(self,input,block):
        self.x          = input
        self.block      = block
        self.activation = block['activation']
        self.filters    = int(block['filters'])
        self.ksize      = int(block['size'])
        self.stride     = int(block['stride'])
        self.index      = block['index']
        self.create_layer_conv2d()

    def create_layer_conv2d(self):
        
        if 'batch_normalize' in self.block:
            bias = False
        else:
            bias = True


        if self.ksize > 1:
            self.x  = ZeroPadding2D(1)(self.x)

        self.x   = Conv2D(self.filters,self.ksize,strides= self.stride,padding = 'valid',name = f"conv_{self.index}",use_bias = bias)(self.x)

        if 'batch_normalize' in self.block:
            self.x  = BatchNormalization(epsilon=0.001,name = f"bn_{self.index}")(self.x)

        if self.activation == 'leaky':
            self.x = LeakyReLU(alpha = 0.1,name = f"leaky_{self.index}")(self.x)
        

class OtherLayers:

    def __init__(self,input,block):
        self.index     = block['index']
        self.x         = input
        self.block     = block


    def UpsamplingLayer(self):
        self.x = UpSampling2D(2)(self.x)
  
    
    def ShortcutLayer(self,prev_layers):
        for layer in prev_layers:
            if layer.index == int(self.block['index']) - abs(int(self.block['from'])):
                self.x = add([layer.x,self.x])
    
                
    def RouteLayer(self,prev_layers):
        if "," in self.block['layers']:
            prev = int(self.block['layers'].split(",")[1])
            for layer in prev_layers:
                if layer.index == prev:
                    self.x = concatenate([self.x,layer.x])
        
        else:
            for layer in prev_layers:
                if layer.index == self.block['index'] - abs(int(self.block['layers'])):
                    self.x = layer.x

        

def create_yolov3(blocks):

    input_image   = Input(shape = (None,None,3))
    layers        = []
    yolov3_layers = []

    for block in blocks:
        if block['type'] == 'convolutional':
            try:
                conv     = Conv2DLayer(x,block)
            except:
                conv     = Conv2DLayer(input_image,block)

            layers.append(conv)
            x            = conv.x


        if block['type'] == 'upsample':
            upsample     = OtherLayers(x,block)
            upsample.UpsamplingLayer()
            x            = upsample.x
            layers.append(upsample)
            

        if block['type'] == 'shortcut':
            shortcut     = OtherLayers(x,block)
            shortcut.ShortcutLayer(layers) 
            x            = shortcut.x
            layers.append(shortcut)
            
    
        if block['type'] == 'route':
            route        = OtherLayers(x,block)
            route.RouteLayer(layers)
            x            = route.x
            layers.append(route)
            

        if block['type'] == 'yolo':
            yolov3_layers.append(x)

        

    model = Model(input_image,yolov3_layers)

    return model




        




    


    






    


    

    



             




    