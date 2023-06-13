import cv2
import os
import numpy as np
from utils.annotations import get_annotations
from utils.data_preprocess import image_augment

import paddle


def get_bbox(gt_bbox, gt_class):
    """
    对于一般的检测任务来说，一张图片上往往会有多个目标物体。
    设置参数 MAX_MUM = 50，即一张图片最多取 50 个真实框。
    如果真实框的数目少于 50 个，则将不足部分的 gt_bbox, gt_class 和 gt_score 的各项数值全设置为 0。
    :param gt_bbox: 真实框
    :param gt_class: 真实框对应的类别
    :return:
    """
    MAX_MUM = 50
    gt_bbox2 = np.zeros((MAX_MUM, 4))
    gt_class2 = np.zeros((MAX_MUM, ))

    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_MUM:
            break
    return gt_bbox2, gt_class2


def get_img_data_from_file(record: dict) -> tuple:
    """
    record is a dict as following,
        record = {
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
    """
    img_file = record['img_file']
    h = record['h']
    w = record['w']
    is_crowd = record['is_crowd']
    gt_class = record['gt_class']
    gt_bbox = record['gt_bbox']
    difficult = record['difficult']

    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检查 h 和 w 是否等于从 img 读取的 h 和 w
    assert img.shape[0] == int(h), \
            f'image height of {img_file} inconsistent in record({h}) and img file({img.shape[0]})'
    assert img.shape[1] == int(w), \
            f'image width of {img_file} inconsistent in record({w}) and img file({img.shape[1]})'

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值，以适应不同大小的图片
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)  # gt_boxes 的格式是 xywh


def get_img_data(record, size=640) -> tuple:
    """
    获取数据
    :param record: 图像的标注信息
    :param size: 图像缩放后的大小
    :return: 图像，真实框，真实框对应的类别，原图像的高宽
    """
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)
    # 图像增强
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, -1))   # [1, 1, 3]
    std = np.array(std).reshape((1, 1, -1))     # [1, 1, 3]
    img = (img / 255. - mean) / std
    img = img.astype('float32').transpose((2, 0, 1))    # [C, H, W]
    return img, gt_boxes, gt_labels, scales


def get_img_size(mode):
    """
    获取一个批次内样本随机缩放的尺寸
    :param mode: train or valid
    :return: 图像大小
    """
    if mode == 'train' or mode == 'valid':
        inds = np.arange(10)
        ii = np.random.choice(inds)
        img_size = 320 + ii * 32
    else:
        img_size = 608
    return img_size


def make_array(batch_data):
    """
    将 list 形式的 batch 数据 转换成 多个 array 构成的 tuple
    :param batch_data: 一个批次的数据
    :return: array 类型的数据
    """
    img_array = np.array([item[0] for item in batch_data], dtype='float32')
    gt_boxes_array = np.array([item[1] for item in batch_data], dtype='float32')
    gt_labels_array = np.array([item[2] for item in batch_data], dtype='int32')
    img_scale_array = np.array([item[3] for item in batch_data], dtype='int32')
    return img_array, gt_boxes_array, gt_labels_array, img_scale_array


def make_test_array(batch_data):
    """
    将 list 形式的 batch 数据 转换成 多个 array 构成的 tuple
    :param batch_data: 一个批次的数据
    :return: array 类型的数据
    """
    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array([item[1] for item in batch_data], dtype='float32')
    img_scale_array = np.array([item[2] for item in batch_data], dtype='int32')
    return img_name_array, img_data_array, img_scale_array  # img_scale_array 是图片的高宽


class TrainDataset(paddle.io.Dataset):
    """
    昆虫数据集类
    """
    def __init__(self, datadir, mode='train'):
        self.datadir = datadir
        self.records = get_annotations(datadir)
        self.img_size = 640  #get_img_size(mode)

    def __getitem__(self, idx):
        record = self.records[idx]
        img, gt_boxes, gt_labels, img_shape = get_img_data(record, size=self.img_size)  # img_shape 是图片原始的高宽

        return img, gt_boxes, gt_labels, np.array(img_shape)

    def __len__(self):
        return len(self.records)


def test_data_loader(datadir, batch_size=10, test_image_size=608, mode='test'):
    """
    测试数据读取器 (测试数据没有 groundtruth 标签)
    :param datadir: 数据路径 (路径到 images 文件夹)
    :param batch_size: 批大小
    :param test_image_size: 图像缩放后的大小
    :param mode:
    :return: 数据读取器
    """
    img_names = os.listdir(datadir)
    def reader():
        batch_data = []
        img_size = test_image_size
        for img_name in img_names:
            # 读取图片并 resize
            file_path = os.path.join(datadir, img_name)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [h, w, c]
            H = img.shape[0]
            W = img.shape[1]
            img = cv2.resize(img, (img_size, img_size))

            # 对图片归一化，并转换通道
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255. - mean) / std
            out_img = out_img.astype('float32').transpose((2, 0, 1))  # [c, h, w]
            img = out_img
            img_shape = [H, W]

            batch_data.append(
                (img_name.split('.'), img, img_shape)
            )
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield  make_test_array(batch_data)
    return reader


if __name__ == '__main__':

    TRAINDIR = '../data/insects/train'
    TESTDIR = '../data/insects/test/images'

    # 实例化数据集类对象
    train_dataset = TrainDataset(TRAINDIR, mode='train')

    # 创建数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2,
                                        drop_last=True)
    test_loader = test_data_loader(TESTDIR)

    img_name, img, img_shape = next(test_loader())  # img_shape 是图片原始的高宽
    print(f'img_name: {img_name}, \nimg.shape: {img.shape}, \nimg_shape.shape: {img_shape.shape}')

