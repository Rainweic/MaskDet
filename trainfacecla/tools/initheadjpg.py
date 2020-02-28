import os
import cv2 as cv
from xml.dom.minidom import parse

"""
读取xml文件 把人脸图像截图并分类
"""

def readXML(xml):
    NoneNum = 0
    print(xml)
    domTree = parse(xml)
    rootNode = domTree.documentElement
    imgName = rootNode.getElementsByTagName("filename")[0].childNodes[0].data
    # imgName = os.path.basename(xml)[:-4] + ".jpg"
    imgName = os.path.join(img_root, imgName)
    print(imgName)
    img = cv.imread(imgName)
    if img is None:
        print("None")
    obj = rootNode.getElementsByTagName("object")
    for o in obj:
        name = o.getElementsByTagName("name")[0].childNodes[0].data
        if name == "face":
            xmin, ymin, xmax, ymax = getBBox(o)
            savePath = nomask_root
        else:
            xmin, ymin, xmax, ymax = getBBox(o)
            savePath = mask_root
        print(xmin, ymin, xmax, ymax)
        headImg = img[ymin:ymax, xmin:xmax]
        saveName = os.path.join(savePath, os.path.basename(imgName[:-4])+"_"+name+".jpg")
        print(saveName)
        cv.imwrite(saveName, headImg)
    print("---------{}--------".format(NoneNum))


def getBBox(node):
    xmin = int(node.getElementsByTagName("xmin")[0].childNodes[0].data)
    ymin = int(node.getElementsByTagName("ymin")[0].childNodes[0].data)
    xmax = int(node.getElementsByTagName("xmax")[0].childNodes[0].data)
    ymax = int(node.getElementsByTagName("ymax")[0].childNodes[0].data)
    return xmin, ymin, xmax, ymax

if __name__ == '__main__':
    data_root = "/Users/rainweic/Downloads/mask/trainfacecla"

    img_root = os.path.join(data_root, "jpg")
    label_root = os.path.join(data_root, "label")
    mask_root = os.path.join(data_root, "mask")
    nomask_root = os.path.join(data_root, "nomask")

    xmls = os.listdir(label_root)
    xmls = [os.path.join(label_root, i) for i in xmls]

    for xml in xmls:
        readXML(xml)