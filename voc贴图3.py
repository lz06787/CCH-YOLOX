'''
从训练集中根据标签获得van、truck和bus
不旋转
贴回原图
即从哪张抠取的，就随机数量贴回哪张

'''

from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil

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


def voc_recs(
    annopath,
    imagesetfile,
    ):

    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath+'/{}.xml'.format(imagename))
        if i % 100 == 0:
            print("Reading annotation for {:d}/{:d}".format(i + 1, len(imagenames)))
    
    return recs


def get_crop_img(annopath, imagesetfile, img, name):
    
    

    img_name = name+'.jpg'

    rec_info = recs[name]

    num = 0
    for obj in rec_info:
        num += 1
        if obj['name'] in ['van','truck','bus']:
    
            
            x1,y1,x2,y2 = obj['bbox']

            # 排除像素太小的crop img
            if (y2-y1)<crop_thre or (x2-x1)<crop_thre:
                continue

            cropImg = img[y1:y2, x1:x2]    # 裁剪【y1,y2：x1,x2】


            if os.path.exists(crop_datasets_path+"/{}".format(obj['name'])) is not True:
                os.makedirs(crop_datasets_path+"/{}".format(obj['name']))
            cv2.imwrite(os.path.join(crop_datasets_path+"/{}".format(obj['name']),'{}_{}_{}.jpg'.format(name,obj['name'],num)),cropImg)



def xml_append(file,name,obj):
    dom = minidom.parse(file)
    root = dom.documentElement
    nobject = dom.createElement('object')
    
    nname = dom.createElement('name')
    tname = dom.createTextNode(name)
    nname.appendChild(tname)

    ndifficult = dom.createElement('difficult')
    tdifficult = dom.createTextNode('0')
    ndifficult.appendChild(tdifficult)

    nbndbox = dom.createElement('bndbox')
    nxmin = dom.createElement('xmin')
    txmin = dom.createTextNode(str(obj['bbox'][0]))
    nxmin.appendChild(txmin)
    nbndbox.appendChild(nxmin)

    nymin = dom.createElement('ymin')
    tymin = dom.createTextNode(str(obj['bbox'][1]))
    nymin.appendChild(tymin)
    nbndbox.appendChild(nymin)

    nxmax = dom.createElement('xmax')
    txmax = dom.createTextNode(str(obj['bbox'][2]))
    nxmax.appendChild(txmax)
    nbndbox.appendChild(nxmax)

    nymax = dom.createElement('ymax')
    tymax = dom.createTextNode(str(obj['bbox'][3]))
    nymax.appendChild(tymax)
    nbndbox.appendChild(nymax)

    nobject.appendChild(nname)
    nobject.appendChild(ndifficult)
    nobject.appendChild(nbndbox)
    
    root.appendChild(nobject)

    dom.writexml()


def write_xml(file, save_file,classes, obj_list):
        tree = ET.parse(file)
        root = tree.getroot()
        for obj in obj_list:
            newobj = ET.Element('object')
            ET.SubElement(newobj, 'name').text = classes
            ET.SubElement(newobj, 'pose').text = 'Unspecified'
            ET.SubElement(newobj, 'truncated').text = '0'
            ET.SubElement(newobj, 'difficult').text = '0'
            bndbox = ET.SubElement(newobj,'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(obj[0])
            ET.SubElement(bndbox, 'ymin').text = str(obj[1])
            ET.SubElement(bndbox, 'xmax').text = str(obj[2])
            ET.SubElement(bndbox, 'ymax').text = str(obj[3])
            root.append(newobj)

        __indent(root)
        tree.write(save_file, encoding='utf-8',xml_declaration=True)


def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():

    trainval = open(os.path.join(save_root_path,'trainval.txt'),'w')
    # 打乱背景图片
    random.shuffle(names)

    num = 0
    for name in names:
        num+=1

        list_total = []
        rec_info = recs[name]
        for obj in rec_info:
            list_total.append(obj['bbox'])



        bg_file_name = os.path.join(bg_datasets_path, name+'.jpg')
        bg_img = Image.open(bg_file_name)
        (bg_w ,bg_h) = bg_img.size
        
        # 创建 label的txt文件
        save_img_name = os.path.join(img_save_path, name+'.jpg')

        xml_file = os.path.join(annopath,'{}.xml'.format(name))
        save_xml_file = os.path.join(label_save_path,'{}.xml'.format(name))



        temp_crop_list = []
        for crop in crop_files:
            if name in crop:
                temp_crop_list.append(crop)

        if len(temp_crop_list) == 0:
            #说明这张图上都是小轿车，那么不复制图片和标签，直接跳过
            continue
        
        # 重写 trainval.txt文件
        trainval.write('{}\n'.format(name))

        # 每张背景上贴的图片数量 0 ~ maxcrop_num_per_img 之间随机一个数
        per_pic_num = random.randint(0,maxcrop_num_per_img)
        print('已完成{}张图片,贴了{}张crop'.format(num, per_pic_num))
        if per_pic_num == 0:
            # 即不贴图，那么就直接复制图片和标签
            shutil.copy(xml_file, save_xml_file)
            shutil.copy(bg_file_name, save_img_name)
            continue

        # 挑选口罩数据集
        #paste_mask_order = random.sample(range(crop_num), per_pic_num)
        paste_mask_order = []
        for i in range(per_pic_num):
            paste_mask_order.append(random.randint(0, len(temp_crop_list)-1))

        
        obj_list = []
        for i in paste_mask_order:
            #* 通过图片名字获得label
            classes = temp_crop_list[i].split('_')[-2]

            mask_file_path = os.path.join(crop_datasets_path,temp_crop_list[i])
            #m_img = Image.open(mask_file_path)
            m_img = cv2.imread(mask_file_path)

            #(m_h0,m_w0,m_d0) = m_img.shape

            m_img = Image.fromarray(cv2.cvtColor(m_img,cv2.COLOR_BGR2RGB))
            # 旋转crop img
            angle = random.randint(0,angle_max)
            #angle = 0
            mask = Image.new('L',m_img.size,255)
            rm_img = m_img.rotate(angle, expand=True)
            rmask = mask.rotate(angle,expand=True)

            rm_img_w, rm_img_h = rm_img.size
            m_w0 = rm_img_w
            m_h0 = rm_img_h

 
            while True:
                # s用来记录已经贴好的图片数量
                try_num = 0

                # 调整口罩图片大小（随机）
                m_w = random.randint(25,100)
                m_h = int(m_w*m_h0/m_w0)

                #m_img=cv2.resize(m_img,(m_w,m_h))

                rm_img = rm_img.resize((m_w, m_h),Image.BICUBIC)
                rmask = rmask.resize((m_w, m_h),Image.BICUBIC)

                # x1 y1 x2 y2 为贴图位置
                x1 = random.randint(0,bg_w)
                y1 = random.randint(0,bg_h)
                x2 = x1+m_w
                y2 = y1+m_h
                if x2>bg_w or y2>bg_h:
                    continue

                paste = True
                for list in list_total:
                    if list == None:
                        continue
                    else:
                        if x1 > list[2] or x2 < list[0] or y1>list[3] or y2 < list[1]:
                            pass 
                        else:
                            paste = False
                
                if paste == False:
                    try_num += 1
                    if try_num == 10:
                        break
                    else:
                        continue
                else:
                    # bg_img.paste(rm_img,(x2,y2))
                    # bg_img.show()

                    bg_img.paste(rm_img,(x1,y1,x2,y2),rmask)
                    #bg_img.show()

                    bg_img.save(save_img_name)

                    list_total.append([x1,y1,x2,y2])

                    obj_list.append([x1,y1,x2,y2])
                    
                    break
        
        write_xml(file=xml_file, save_file=save_xml_file, classes=classes, obj_list=obj_list)
        
    trainval.close()



def get_names(path):
    
    txt = open(path,'r')
    for line in txt:
        names.append(line.strip('\n'))
    
    return names

if __name__ == "__main__":

    # 是否重新根据标签获取van、truck和bus
    IF_Crop = True 
    
    # 是否贴图
    IF_Paste = True

    # 设置保存裁剪图像的阈值，太小的不保存
    crop_thre = 25

    # 每张背景图的最多贴图数量
    maxcrop_num_per_img = 10
    
    # 贴图时最大旋转角度，0为不旋转，最大360
    angle_max = 0
    
    # 背景图像文件夹
    bg_datasets_path = 'datasets/train/JPEGImages'
    # 裁剪图像保存文件夹
    crop_datasets_path = 'datasets/crop_img'
    # 背景图像的标签文件夹
    annopath = 'datasets/train/Annotations'
    # voc文件的imageset文件夹，路径需到txt文件
    imagesetfile = 'datasets/train/trainval.txt'

    # 贴图保存根目录
    save_root_path = 'number_paste'
    # 贴图保存文件夹
    img_save_path = 'number_paste/JPEGImages'
    # 贴图的标签保存文件夹
    label_save_path = 'number_paste/Annotations'

    if os.path.exists(save_root_path) is not True:
        os.makedirs(save_root_path)
    if os.path.exists(crop_datasets_path) is not True:
        os.makedirs(crop_datasets_path)
    if os.path.exists(img_save_path) is not True:
        os.makedirs(img_save_path)
    if os.path.exists(label_save_path) is not True:
        os.makedirs(label_save_path)

    recs = voc_recs(annopath,imagesetfile)
    
    names = []
    names = get_names(imagesetfile)
    name_num = 0

    '''
    get_crop_img 通过标签获取van、truck和bus
    '''
    if IF_Crop:
        for name in names:
            name_num+=1
            img = cv2.imread(os.path.join(bg_datasets_path,name+'.jpg'))
            get_crop_img(annopath=annopath, imagesetfile=imagesetfile, img=img, name=name)
            print(name_num)


    crop_files = os.listdir(crop_datasets_path)
    crop_num = len(crop_files)
    bg_files = os.listdir(bg_datasets_path)
    bg_num = len(bg_files)

    '''
    main 贴图函数
    '''
    if IF_Paste:
        main()
