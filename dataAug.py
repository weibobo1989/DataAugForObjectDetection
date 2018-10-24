import os
import cv2
import xml.dom.minidom
from xml.dom.minidom import Document
import math
import xml.dom.minidom as DOC
import xml.etree.ElementTree as ET
import numpy as np
import random
from skimage.util import random_noise
from skimage import exposure


# 获取路径下所有文件的完整路径，用于读取文件用
def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles



# 读取xml文件，xmlfile参数表示xml的路径
def readXml(xmlfile):
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    sizelist = annotation.getElementsByTagName('size')  # [<DOM Element: filename at 0x381f788>]
    heights = sizelist[0].getElementsByTagName('height')
    height = int(heights[0].childNodes[0].data)
    widths = sizelist[0].getElementsByTagName('width')
    width = int(widths[0].childNodes[0].data)
    depths = sizelist[0].getElementsByTagName('depth')
    depth = int(depths[0].childNodes[0].data)
    objectlist = annotation.getElementsByTagName('object')
    bboxes = []
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        class_label = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')[0]
        x1_list = bndbox.getElementsByTagName('xmin')
        x1 = int(float(x1_list[0].childNodes[0].data))
        y1_list = bndbox.getElementsByTagName('ymin')
        y1 = int(float(y1_list[0].childNodes[0].data))
        x2_list = bndbox.getElementsByTagName('xmax')
        x2 = int(float(x2_list[0].childNodes[0].data))
        y2_list = bndbox.getElementsByTagName('ymax')
        y2 = int(float(y2_list[0].childNodes[0].data))
        # 这里我box的格式【xmin，ymin，xmax，ymax，classname】
        bbox = [x1, y1, x2, y2, class_label]
        bboxes.append(bbox)
    return bboxes, width, height, depth

def parse_xml(xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''

    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(float(box[0].text))
        y_min = int(float(box[1].text))
        x_max = int(float(box[2].text))
        y_max = int(float(box[3].text))
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

# 写xml文件，参数中tmp表示路径，imgname是文件名（没有尾缀）ps有尾缀也无所谓
def writeXml(tmp, imgname, w, h, d, bboxes):
    doc = Document()
    # owner
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    # owner
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder_txt = doc.createTextNode("VOC2007_LISA")
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    annotation.appendChild(filename)
    filename_txt = doc.createTextNode(imgname)
    filename.appendChild(filename_txt)
    # ones#
    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode("My Database")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode("VOC2007_LISA")
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)
    # owner
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)

    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)
    # onee#
    # twos#
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)


    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
    # twoe#
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("0")
    segmented.appendChild(segmented_txt)

    for bbox in bboxes:
        # 'go'数据量足够，不需要增强
        if bbox[4].lower() == 'go':
            continue
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[4]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode("0")
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('bndbox')
        object_new.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(float(int(bbox[0]))))
        xmin.appendChild(xmin_txt)

        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(float(int(bbox[1]))))
        ymin.appendChild(ymin_txt)

        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(float(int(bbox[2]))))
        xmax.appendChild(xmax_txt)

        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(float(int(bbox[3]))))
        ymax.appendChild(ymax_txt)

        print(bbox[4], float(int(bbox[0])), float(int(bbox[1])), float(int(bbox[2])), float(int(bbox[3])))

    tempfile = tmp + "/%s.xml" % imgname
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    return

def generate_xml(img_name,coords,img_size,out_root_path):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The Tianchi Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:

        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(os.path.jpin(out_root_path, img_name),'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()


#加噪声
def addNoise(img, bboxes):
    '''
    输入:
        img:图像array
    输出:
        加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
    '''
    # random.seed(int(time.time()))
    # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
    return random_noise(img, mode='gaussian', clip=True)*255, bboxes

# 调整亮度
def changeLight(img, bboxes):
    # random.seed(int(time.time()))
    flag = random.uniform(0.5, 1.5) #flag>1为调暗,小于1为调亮
    return exposure.adjust_gamma(img, flag), bboxes

# 平移图像
def shift_pic_bboxes(img, bboxes):
    '''
    参考:https://blog.csdn.net/sty945/article/details/79387054
    平移后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        shift_img:平移后的图像array
        shift_bboxes:平移后的bounding box的坐标list
    '''
    # ---------------------- 平移图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    x_min = w  # 裁剪后的包含所有目标框的最小的框
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    d_to_left = x_min  # 包含所有目标框的最大左移动距离
    d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
    d_to_top = y_min  # 包含所有目标框的最大上移动距离
    d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

    x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
    y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

    M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # ---------------------- 平移boundingbox ----------------------
    shift_bboxes = list()
    for bbox in bboxes:
        shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y, bbox[4]])

    return shift_img, shift_bboxes

# 裁剪
def crop_img_bboxes(img, bboxes):
    '''
    裁剪后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        crop_img:裁剪后的图像array
        crop_bboxes:裁剪后的bounding box的坐标list
    '''
    # ---------------------- 裁剪图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    x_min = w  # 裁剪后的包含所有目标框的最小的框
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    d_to_left = x_min  # 包含所有目标框的最小框到左边的距离
    d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
    d_to_top = y_min  # 包含所有目标框的最小框到顶端的距离
    d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

    # 随机扩展这个最小框
    crop_x_min = int(x_min - random.uniform(0, d_to_left))
    crop_y_min = int(y_min - random.uniform(0, d_to_top))
    crop_x_max = int(x_max + random.uniform(0, d_to_right))
    crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

    # 随机扩展这个最小框 , 防止别裁的太小
    # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
    # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
    # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
    # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

    # 确保不要越界
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)

    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # ---------------------- 裁剪boundingbox ----------------------
    # 裁剪后的boundingbox坐标计算
    crop_bboxes = list()
    for bbox in bboxes:
        crop_bboxes.append([bbox[0] - crop_x_min, bbox[1] - crop_y_min, bbox[2] - crop_x_min, bbox[3] - crop_y_min, bbox[4]])

    return crop_img, crop_bboxes



#数据增强
def data_augmentation(fn, imgs_path, annos_path, imgs_new_path, annos_new_path):
    # 返回每一张原图的路径
    imgs_path_list = GetFileFromThisRootDir(imgs_path)
    #对所有图像操作
    for num, img_path in enumerate(imgs_path_list):
        # 打印处理进度
        print('processing %d of %d ' % (num+1, len(imgs_path_list)))
        #读入图像
        img = cv2.imread(img_path)
        # 得到原图的名称
        file_name = os.path.basename(os.path.splitext(img_path)[0])
        # 读取anno标签数据，返回相应的信息
        anno = os.path.join(annos_path, '%s.xml' % file_name)
        gts = parse_xml(anno)

        # 保存处理后图片的路径
        save_pic_path = '_'.join([imgs_new_path, fn.__name__])
        if not os.path.isdir(save_pic_path):
            os.makedirs(save_pic_path)

        # 保存处理后label的路径
        save_ann_path = annos_new_path  # os.path.join(annos_new_path, fn.__name__)
        if not os.path.isdir(save_ann_path):
            os.makedirs(save_ann_path)

        # 循环处理数次
        for i in range(2):
            #处理图像(需要调整bbox)
            img_new, gts_new = fn(img, gts)
            ## 处理图像(不需要调整bbox)
            ## img_new, gts_new = fn(img), gts

            # 得到处理后的图像的高、宽、深度，用于书写xml
            H, W, D = img_new.shape
            #img_size = img_new.shape


            # 保存处理后图像
            # print('save %s_%s_%d.jpg' % (file_name, fn.__name__, i))
            cv2.imwrite(os.path.join(save_pic_path, '%s_%s_%d.jpg' % (file_name, fn.__name__, i)), img_new)

            #写annotation.xml:
            # print('save %s_%s_%d.xml' % (file_name, fn.__name__, i))
            writeXml(save_ann_path, '%s_%s_%d' % (file_name, fn.__name__, i), W, H, D, gts_new)
            #generate_xml('%s_%s' % (file_name, fn.__name__), gts_new, img_size, annos_new_path)


if __name__ == '__main__':
    # voc路径
    root = '/home/kevin/Documents/traffic_light/LISA/traffic_light/dayTrain'
    img_raw_dir = root + '/warningJpegsNew'
    annos_path = root + '/warningAnnotationsNew'

    # 返回每一张原图的路径
    # imgs_path = GetFileFromThisRootDir(img_dir)

    # 存储新的anno位置
    annos_new_path = root + '/warningAnnotationsNew/0mergeProcess'
    # if not os.path.isdir(annos_new_path):
    #     os.makedirs(annos_new_path)

    # 存储新图片保存的位置
    imgs_new_path = root + '/warningJpegsNew/0mergeProcess'
    # if not os.path.isdir(imgs_new_path):
    #     os.makedirs(imgs_new_path)

    img_dirs = os.listdir(img_raw_dir)

    fn_dic = {'addNoise': addNoise, 'changeLight': changeLight,
              'shift_pic_bboxes': shift_pic_bboxes, 'crop_img_bboxes': crop_img_bboxes}

    for i in range(len(img_dirs)):
        for j in range(i+1, len(img_dirs)):
            imgs_path = os.path.join(img_raw_dir, img_dirs[i])
            annos_path = os.path.join(annos_path, img_dirs[i])
            imgs_new_path = os.path.join(imgs_new_path, img_dirs[i])
            annos_new_path = os.path.join(annos_new_path, img_dirs[i])

           
            data_augmentation(fn_dic[img_dirs[j]], imgs_path, annos_path, imgs_new_path, annos_new_path)




