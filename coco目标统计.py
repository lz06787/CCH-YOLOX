from matplotlib import pyplot as plt
import tkinter
import matplotlib
from torchvision.datasets import CocoDetection
matplotlib.use('TkAgg')

import numpy as np
basedir = '../datasets/coco'
c = CocoDetection(basedir, '../datasets/coco/annotations/instances_test.json')
roidb = c.load(add_gt=True, add_mask=True)
print("#Images:", len(roidb))
area_list = []
def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList, 100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin, Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin, Ymax)
    plt.title(Title)
    plt.show()
for i in range(len(roidb)):
    boxes = roidb[i]["boxes"]
    # 变形
    #print(i)
    # print(boxes)
    for box in boxes:
        area = (box[2]-box[0]) * (box[3] - box[1])
        area_list.append(area)

#draw_hist(area_list, 'AreasList', 'Area', 'number', 0, 640000, 0.0, 5000)

data = np.array(area_list)
print('len(data)', len(data))
num3 = np.where(data<300)
print('len(num16)', len(num3[0]))

num16 = np.where(data<512)
print('len(num16)', len(num16[0]))

num32 = np.where(data<1024)
print('len(num32)', len(num32[0]))

num64 = np.where(data<64*64)
print('len(num64)', len(num64[0]))

num128 = np.where(data<128*128)
print('len(num128)', len(num128[0]))