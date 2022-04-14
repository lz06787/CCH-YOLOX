
import xml.etree.ElementTree as ET
import pickle
from xml.dom import minidom

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        #obj_struct["pose"] = obj.find("pose").text
        #obj_struct["truncated"] = int(obj.find("truncated").text)
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


def xml_append(file,name='0',obj=[0,0,0,0]):
    dom = minidom.parse(file)
    #dom = minidom.Document()
    root = dom.documentElement
    #root = dom.createElement('annotation')
    #dom.appendChild(root)

    nobject = dom.createElement('object')
    
    nname = dom.createElement('name')
    tname = dom.createTextNode(name)
    nname.appendChild(tname)

    ndifficult = dom.createElement('difficult')
    tdifficult = dom.createTextNode('0')
    ndifficult.appendChild(tdifficult)

    nbndbox = dom.createElement('bndbox')
    nxmin = dom.createElement('xmin')
    txmin = dom.createTextNode(str(obj[0]))
    nxmin.appendChild(txmin)
    nbndbox.appendChild(nxmin)

    nymin = dom.createElement('ymin')
    tymin = dom.createTextNode(str(obj[1]))
    nymin.appendChild(tymin)
    nbndbox.appendChild(nymin)

    nxmax = dom.createElement('xmax')
    txmax = dom.createTextNode(str(obj[2]))
    nxmax.appendChild(txmax)
    nbndbox.appendChild(nxmax)

    nymax = dom.createElement('ymax')
    tymax = dom.createTextNode(str(obj[3]))
    nymax.appendChild(tymax)
    nbndbox.appendChild(nymax)

    nobject.appendChild(nname)
    nobject.appendChild(ndifficult)
    nobject.appendChild(nbndbox)
    
    root.appendChild(nobject)

    
    with open("0000001.xml", "w", encoding="utf-8") as f:
        # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
        # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
        dom.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")



def write_xml(file,obj=[0,0,0,0]):
        tree = ET.parse(file)
        root = tree.getroot()

        newobj = ET.Element('object')
        
        ET.SubElement(newobj, 'name').text = '00000'
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
        tree.write("0000001.xml", encoding='utf-8',xml_declaration=True)


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


#parse_rec('0000001.xml')
#xml_append(file='datasets/val/Annotations/0000001_02999_d_0000005.xml')
write_xml(file='datasets/val/Annotations/0000001_02999_d_0000005.xml')

