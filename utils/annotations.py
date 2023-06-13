import os
import numpy as np
import xml.etree.ElementTree as ET

INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus',
                'acuminatus', 'armandi', 'coleoptera', 'linnaeus']


def get_insect_names() -> dict:
    """
    将昆虫的类别名字（字符串）转化成数字表示的类别。
    return a dict, as following,
        {'Boerner': 0,
         'Leconte': 1,
         'Linnaeus': 2,
         'acuminatus': 3,
         'armandi': 4,
         'coleoptera': 5,
         'linnaeus': 6
        }
    It can map the insect name into an integer label.
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id


def get_annotations(datadir: str) -> list:
    '''
    获取所有图片的标注信息。
    :param datadir: 数据路径
    :return:
    '''
    # 获取类别名字转数字的字典
    cname2cid = get_insect_names()
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))  # ['100.xml', '102.xml', ...]
    # 所有图片的标注信息的列表
    records = []
    ct = 0
    for fname in filenames:
        if 'xml' not in fname:
            continue
        fid = fname.split('.')[0]  # ['100', '102', ...]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)  # 标注文件的路径
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')  # 图片的路径
        tree = ET.parse(fpath)  # 解析 xml 文件

        # 获取 image id
        if tree.find('id') is None:
            img_id = np.array([ct])
        else:
            img_id = np.array(int[tree.find('id ').text])

        # 获取所有目标物体的标注
        objs = tree.findall('object')
        img_w = float(tree.find('size').find('width').text)  # 图片宽
        img_h = float(tree.find('size').find('height').text)  # 图片高
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)  # 真实框坐标
        gt_class = np.zeros((len(objs),), dtype=np.int32)  # 真实标签
        is_crowd = np.zeros((len(objs),), dtype=np.int32)  # 是否为拥挤场景
        difficult = np.zeros((len(objs),), dtype=np.int32)  # 识别是否困难

        for i, obj in enumerate(objs):
            cname = obj.find('name').text  # 目标类别名称
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            # 获取真实框的坐标
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)

            x1 = max(0, x1)  # 真实框左上角横坐标不能小于0
            y1 = max(0, y1)  # 真实框左上角纵坐标不能小于0
            x2 = min(img_w - 1, x2)  # 真实框右下角横坐标不能大于图片的宽度
            y2 = min(img_h - 1, y2)  # 真实框右下角纵坐标不能大于图片的高度
            # 这里使用 xywh 格式来表示目标物体的真实框
            gt_bbox[i] = [(x1 + x2) / 2., (y1 + y2) / 2., x2 - x1 + 1., y2 - y1 + 1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        # 存储每个图片的标注信息
        voc_rec = {
            'img_file': img_file,
            'img_id': img_id,
            'h': img_h,
            'w': img_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
        }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


if __name__ == '__main__':

    TRAINDIR = '../data/insects/train'
    cname2cid = get_insect_names()
    records = get_annotations(TRAINDIR)
    print(f'records length: {len(records)}')
    print(records[0])
