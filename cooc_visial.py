from pycocotools.coco import COCO
 
dataDir='../datasets/coco'
dataType='trainval'  # trainval or test
#dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir, dataType)
 
# initialize COCO api for instance annotations
coco=COCO(annFile)
 
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
anns = coco.loadAnns(coco.getAnnIds())
bboxes = [box["bbox"] for box in anns]
cat_nms=[cat['name'] for cat in cats]
print('number of categories: ', len(cat_nms))
print('COCO categories: \n', cat_nms)

small = 0
median = 0
large = 0

for bbox in bboxes:
    # x0 = bbox[0]
    # y0 = bbox[1]
    # x1 = bbox[2]
    # y1 = bbox[3]
    # w = abs(int(x0)-int(x1))
    # h = abs(int(y0)-int(y1))
    w = bbox[2]
    h = bbox[3]
    if w*h < 32*32:
        small += 1
    elif w*h > 96*96:
        large += 1
    else:
        median += 1

print('large: ', large)   
print('median: ', median)
print('small: ', small)



# 统计各类的图片数量和标注框数量
for cat_name in cat_nms:
    catId = coco.getCatIds(catNms=cat_name)     # 1~90
    imgId = coco.getImgIds(catIds=catId)        # 图片的id  
    annId = coco.getAnnIds(catIds=catId)        # 标注框的id
    
    print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))
 
