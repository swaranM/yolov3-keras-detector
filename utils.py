import cv2
import numpy as np
import random
from dataclasses import replace


def cfgparser(cfg):

    blocks = [] 
    file  = open(cfg,'r')
    lines = file.readlines()
    lines = [i.strip() for i in lines]
    lines = [i for i in lines if i]
    lines = list(filter(lambda x: "#" not in x,lines))
    layer_index = 0


    for i in range(len(lines)):
        line  = lines[i]

        if '[' and ']' in line:
            block = {}
            if 'net' not in line:
                block['index'] = layer_index 
                layer_index += 1
            block['type'] = line.replace("[","").replace("]","").strip()
            continue

        block[line.split("=")[0].strip()] = line.split("=")[1].strip()

        if i == len(lines) - 1:
            blocks.append(block)
            continue
        elif '[' and ']' in lines[i+1]:
             blocks.append(block)
        
    return blocks[0],blocks[1:]


def get_letterbox_image(inp_img,target_size):

    img_w   , img_h    = inp_img.shape[1], inp_img.shape[0]
    target_h,target_w  = target_size
    r                  = min(target_w/img_w,target_h/img_h)
    new_w              = int(img_w * r)
    new_h              = int(img_h * r)

    img                = cv2.resize(inp_img,(new_w,new_h),interpolation= cv2.INTER_CUBIC)
    dh                 = target_h - new_h
    dw                 = target_w - new_w

    top, bottom        = dh//2, dh - dh//2
    left, right        = dw//2, dw - dw//2
    new_im             = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
    
    return new_im


def image_resize(inp_img,h,w):

    img_w, img_h = inp_img.shape[1], inp_img.shape[0]
    new_w = int(img_w * (w/img_w))
    new_h = int(img_h * (h/img_h))
    resized_img = cv2.resize(inp_img,(new_w,new_h),interpolation= cv2.INTER_CUBIC)

    return resized_img


def load_weights(blocks,model,weights):

    fp = open(weights,"rb")
    header  = np.fromfile(fp,dtype = np.int32,count=5)
    weights = np.fromfile(fp, dtype = np.float32)
    idx = 0
    for block in blocks:
        if block['type'] == 'convolutional':
            if 'batch_normalize' in block:
                bn_layer     = model.get_layer("bn_{}".format(block['index']))
                size = np.prod(bn_layer.get_weights()[0].shape)
                beta    = weights[idx:idx+size]
                idx     = idx + size
                std_dev = weights[idx:idx+size]
                idx     = idx + size
                mean    = weights[idx:idx+size]
                idx     = idx + size
                variance= weights[idx:idx+size]
                idx     = idx + size
                bn_layer.set_weights([std_dev,beta,mean,variance])
            else:
                conv_layer   = model.get_layer("conv_{}".format(block['index']))
                conv_weights = conv_layer.get_weights()
                kernel_size  = np.prod(conv_weights[0].shape)
                bias_size    = np.prod(conv_weights[1].shape)
                bias         = weights[idx:idx+bias_size]
                idx          = idx + bias_size
                kernel       = weights[idx:idx+kernel_size]
                idx          = idx + kernel_size
                kernel       = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel       = kernel.transpose([2,3,1,0])
                conv_layer.set_weights([kernel,bias])
                continue
            
            conv_layer   = model.get_layer("conv_{}".format(block['index']))
            conv_weights = conv_layer.get_weights()
            kernel_size  = np.prod(conv_layer.get_weights()[0].shape)
            kernel       = weights[idx:idx+kernel_size]
            idx          = idx + kernel_size
            kernel       = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel       = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])

        else:
            continue

    return model

def get_labels(fname,blocks):

    with open(fname,"r+") as fp:
        lines      = fp.readlines()
        label_dict = {i:j.strip().capitalize() for i,j in enumerate(lines)}
        colors     = {i:(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in label_dict}
        fp.close()

    for block in blocks:
        if 'yolo' in block['type']:
            no_of_cls = int(block['classes'])

    return label_dict,no_of_cls,colors









