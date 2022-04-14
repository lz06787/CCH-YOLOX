'''
用于可视化xml文件标签的各类别数量数量，需要修改存储xml标签的位置：xml_path
'''

# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as ET
import numpy as np
#np.set_printoptions(suppress=True, threshold=np.nan)
import matplotlib
from PIL import Image
import sys

#========================#

xml_path= 'number_paste/Annotations/'

#========================#
 
def parse_obj(xml_path, filename):
    tree=ET.parse(xml_path+filename)
    objects=[]
    for obj in tree.findall('object'):
        obj_struct={}
        obj_struct['name']=obj.find('name').text
        objects.append(obj_struct)
    return objects
 
 
def read_image(image_path, filename):
    im=Image.open(image_path+filename)
    W=im.size[0]
    H=im.size[1]
    area=W*H
    im_info=[W,H,area]
    return im_info
 
 
if __name__ == '__main__':
    # 打印完整numpy数组
    np.set_printoptions(threshold=sys.maxsize)

    filenamess=os.listdir(xml_path)
    filenames=[]
    for name in filenamess:
        name=name.replace('.xml','')
        filenames.append(name)
    recs={}
    obs_shape={}
    classnames=[]
    num_objs={}
    obj_avg={}
    for i,name in enumerate(filenames):
        recs[name]=parse_obj(xml_path, name+ '.xml' )
    for name in filenames:
        for object in recs[name]:
            print(object)
            if object['name'] not in num_objs.keys():
                num_objs[object['name']]=1
            else:
                num_objs[object['name']]+=1
            if object['name'] not in classnames:
                classnames.append(object['name'])
    for name in classnames:
        print('{}:{}个'.format(name,num_objs[name]))
    print('信息统计算完毕。')

    print(name)
    print(num_objs[name])
    print(classnames)
    import matplotlib.pyplot as plt  #约定俗成的写法plt
    name_list = []  
    num_list = []  
    for name in classnames:
        name_list.append(name)
        num_list.append(num_objs[name])
    plt.bar(range(len(num_list)), num_list,tick_label=name_list)  
    plt.show() 