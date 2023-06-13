import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from utils.box_utils import multi_box_iou_xywh, box_crop


def random_distort(img: np.ndarray) -> np.ndarray:
    """
    随机改变亮暗、对比度和颜色等
    :param img: 待增强的图像
    :return: 增强后的图像
    """

    # 随机改变亮度
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    # 随机改变对比度
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    # 随机改变颜色
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)

    return img


def random_expand(img: np.ndarray,
                  gt_boxes: list,
                  max_ratio: float = 4.,
                  fill=None,
                  keep_ratio: bool = True,
                  thresh: float = 0.5) -> tuple:
    """
    随机填充
    :param img: 原图像
    :param gt_boxes: 真实框
    :param max_ratio: 最大填充比率
    :param fill: 填充图像时使用的颜色，其默认值为 None, 即黑色
    :param keep_ratio: 宽高保持比例
    :param thresh: 控制是否进行填充的概率阈值，其默认值为 0.5
    :return: 增强后的图像, 新的真实框
    """
    if random.random() > thresh:
        return img, gt_boxes

    if max_ratio < 1.0:
        return img, gt_boxes

    h, w, c = img.shape
    # 生成随机填充比例
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    # 根据比例生成填充后图像的宽度 ow 和高度 oh，并计算随机的横向偏移量 off_x 和纵向偏移量 off_y
    ow = int(w * ratio_x)
    oh = int(h * ratio_y)
    off_x = random.randint(0, ow - w)
    off_y = random.randint(0, oh - h)

    out_img = np.zeros((oh, ow, c))
    # 如果设定了 fill 并且其长度等于 c，则将零矩阵的每个通道都填充为 fill 中对应的数值的 255.0 倍
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0

    # 使用原始图像 img 来填充 out_img
    out_img[off_y:off_y + h, off_x:off_x + w, :] = img
    # 将标注框的位置信息进行相应的调整
    gt_boxes[:, 0] = ((gt_boxes[:, 0] * w) + off_x) / float(ow)
    gt_boxes[:, 1] = ((gt_boxes[:, 1] * h) + off_y) / float(oh)
    gt_boxes[:, 2] = gt_boxes[:, 2] * w / float(ow)
    gt_boxes[:, 3] = gt_boxes[:, 3] * h / float(oh)

    return out_img.astype('uint8'), gt_boxes


def random_crop(img,
                gt_boxes,
                gt_labels,
                scales=[0.3, 1.0],
                max_ratio=2.0,
                constraints=None,
                max_trial=50) -> tuple:
    """
    随机裁剪
    :param img: 原图像
    :param gt_boxes: 真实框
    :param gt_labels: 真实框对应的类别
    :param scales:
    :param max_ratio:
    :param constraints: IoU 的约束条件
    :param max_trial: 每个约束条件最大的尝试次数
    :return: 增强后的图像, 新的真实框, 新的真实框对应的类别
    """
    if len(gt_boxes) == 0:
        return img, gt_boxes

    if not constraints:
        constraints = [(0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = img.size  # Image 读取的图片是 w, h 形式
    # 存储各种裁剪框， xywh 形式(这里的 xy 是裁剪框的左上角的坐标)
    crops = [(0, 0, w, h)]  # (0, 0, w, h) 相当于裁剪原图

    # 对于每个约束条件，进行最多 max_trial 次的尝试，生成一个随机的裁剪区域 crop_box
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale),
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            # 构建裁剪框，xywh 形式(这里的 xy 是裁剪框的中心点的坐标)，并从像素值坐标转换为归一化坐标
            crop_box = np.array([[(crop_x + crop_w / 2.) / w,
                                  (crop_y + crop_h / 2.) / h,
                                  crop_w / float(w),
                                  crop_h / float(h)]])

            iou = multi_box_iou_xywh(crop_box, gt_boxes)
            # 把 iou 限制在一个区间，尽可能地裁剪到所有的目标
            if min_iou <= iou.min() and iou.max() <= max_iou:
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    # 随机挑选一个裁剪框来对图片进行裁剪
    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        # 裁剪后的边框，裁剪后的标签，裁剪后的边框的数目
        crop_boxes, crop_labels, box_num = box_crop(gt_boxes, gt_labels, crop, (w, h))
        # 裁剪后的边界框的数目为 0，那就没必要裁剪了
        if box_num == 0:
            continue
        # 对图片裁剪后，将其缩放回原图片的大小，缩放算法为 Image.LANCZOS
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    # 最坏的情况：每一个 iou 限制区间尝试 50 次都没有随机出有效的裁剪框。则直接返回原图的数据
    img = np.asarray(img)
    return img, gt_boxes, gt_labels


def random_interp(img, size, interp=None) -> np.ndarray:
    """
    随机缩放
    :param img: 待缩放的图像
    :param size: 缩放后的图像大小
    :param interp: 插值方法
    :return: 增强后的图像
    """
    # 插值方法
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ]
    if not interp or interp not in interp_method:
        interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    img_scale_x = size / float(w)
    img_scale_y = size / float(h)
    img = cv2.resize(img, None, None, fx=img_scale_x, fy=img_scale_y, interpolation=interp)
    return img


def random_flip(img, gt_boxes, thresh=0.5) -> tuple:
    """
    随机水平翻转
    :param img: 原图像
    :param gt_boxes: 真实框
    :param thresh: 水平翻转的概率
    :return: 增强后的图像, 新的真实框
    """
    if random.random() > thresh:
        img = img[:, ::-1, :]
        gt_boxes[:, 0] = 1.0 - gt_boxes[:, 0]
    return img, gt_boxes


def shuffle_gtboxes(gt_boxes, gt_labels):
    """
    随机打乱真实框的排列顺序
    :param gt_boxes: 真实框
    :param gt_labels: 真实框对应的类别
    :return: 新的真实框，新的真实框对应的类别
    """
    # 将 gt_boxes 和 gt_labels 按列方向拼接起来
    gt = np.concatenate([gt_boxes, gt_labels[:, np.newaxis]], axis=1)  # shape: [N, 5]
    idx = np.arange(gt.shape[0])  # [0, 1, ..., gt.shape[0]], shape: [N, ]
    np.random.shuffle(idx)
    gt = gt[idx, :]
    return gt[:, :4], gt[:, 4]


def image_augment(img, gt_boxes, gt_labels, size, means=None):
    """
    图像增广方法汇总
    :param img: 原图像
    :param gt_boxes: 真实框
    :param gt_labels: 真实框对应的类别
    :param size: 图像缩放后的大小
    :param means: 填充图像时使用的颜色，其默认值为 None, 即黑色
    :return: 增强后的图像，新的真实框，新的真实框对应的类别
    """
    # 随机改变亮暗、对比度和颜色等
    img = random_distort(img)
    # 随机填充
    img, gt_boxes = random_expand(img, gt_boxes, fill=means)
    # 随机裁剪
    img, gt_boxes, gt_labels = random_crop(img, gt_boxes, gt_labels)
    # 随机缩放
    img = random_interp(img, size)
    # 随机水平翻转
    img, gt_boxes = random_flip(img, gt_boxes)
    # 随机打乱真实框的排列顺序
    gt_boxes, gt_labels = shuffle_gtboxes(gt_boxes, gt_labels)
    return img.astype('float32'), gt_boxes.astype('float32'), gt_labels.astype('int32')


def visualize(img_src, img_enhance):
    """
    可视化函数，用于对比原图和图像增强的效果
    :param img_src: 原图像
    :param img_enhance: 增强后的图像
    :return:
    """
    # 图像可视化
    plt.figure(num=2, figsize=(6, 12))
    plt.subplot(1, 2, 1)
    plt.title('Src Image', color='#0000FF')
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(img_src)  # 显示原图片

    plt.subplot(1, 2, 2)
    plt.title('Enhance Image', color='#0000FF')
    plt.axis('off')  # 不显示坐标轴
    plt.imshow(img_enhance)  # 显示增强图片


