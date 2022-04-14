import argparse
import os
import time
from unicodedata import category
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np
import xml.etree.ElementTree as ET
import pickle


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def visual(ann_name, img, filename):
    
    
    rec_info = parse_rec(filename=ann_name)
    
    gt_bboxes = []
    for i in rec_info:
        gt_bboxes.append(i['bbox'])
    

    for i in range(len(gt_bboxes)):
            
            gt_box = gt_bboxes[i]
        
            x0 = int(gt_box[0])
            y0 = int(gt_box[1])
            x1 = int(gt_box[2])
            y1 = int(gt_box[3])

            color = (0,0,255)
        
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)           
            #cv2.imwrite(os.path.join())
    cv2.imshow('img',img)
    cv2.waitKey(0)

if __name__ == "__main__":
    annopath = 'number_paste/label'
    imgpath = 'number_paste/image'

    annolist = os.listdir(annopath)
    for file in annolist:
        img_name = os.path.join(imgpath, file.strip('.xml')+'.jpg')
        ann_name = os.path.join(annopath,file)
        img = cv2.imread(img_name)
        visual(ann_name=ann_name, img=img, filename=file)
    