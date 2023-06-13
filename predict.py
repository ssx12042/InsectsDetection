import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paddle

from net.YOLOv3 import YOLOv3
from utils.annotations import INSECT_NAMES
from utils.box_utils import draw_rectangle, multiclass_nms

from datasets.InsectsDataset import make_test_array


def single_data_loader(file_path, test_image_size=608, mode='test'):
    """
    加载测试用的单张图片，测试数据没有 groundtruth 标签
    :param file_path:
    :param test_image_size:
    :param mode:
    :return:
    """
    batch_size = 1
    def reader():
        batch_data = []
        img_size = test_image_size

        img = cv2.imread(file_path)  # shape: [H, W, C]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H = img.shape[0]
        W = img.shape[1]
        img = cv2.resize(img, (img_size, img_size))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))

        out_img = (img / 255. - mean) / std
        out_img = out_img.astype('float32').transpose((2, 0, 1))  # shape: [C, H, W]

        img = out_img
        img_shape = [H, W]
        img_name = file_path.split('/')[-1]

        batch_data.append((img_name, img, img_shape))
        if len(batch_data) == batch_size:
            yield make_test_array(batch_data)
            batch_data = []

    return reader


def draw_results(result, file_path, draw_thresh=0.5):
    """
    绘制预测结果
    :param result:
    :param file_path:
    :param draw_thresh:
    :return:
    """
    plt.figure(figsize=(10, 10))
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    currentAxis = plt.gca()
    colors = ['r', 'g', 'b', 'k', 'y', 'c', 'purple']
    for item in result:
        box = item[2:6]
        label = int(item[0])
        name = INSECT_NAMES[label]
        if item[1] > draw_thresh:
            draw_rectangle(currentAxis=currentAxis,
                           bbox=box,
                           edgecolor=colors[label])
            plt.text(box[0], box[1], name, fontsize=12, color=colors[label])


if __name__ == '__main__':

    ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    VALID_THRESH = 0.01
    NMS_TOPK = 400
    NMS_POSK = 100
    NMS_THRESH = 0.45

    NUM_CLASSES = 7

    image_path = 'data/insects/test/images/2246.jpeg'
    # 数据读取器
    test_loader = single_data_loader(file_path=image_path, mode='test')

    params_file_path = 'checkpoint/yolo_epoch40'

    model = YOLOv3(num_classes=NUM_CLASSES)
    model_state_dictt = paddle.load(params_file_path)
    model.load_dict(model_state_dictt)
    model.eval()

    total_results = []
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data  # img.shape: [N, 3, 608, 608]
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale_data)

        outputs = model(img)  # P0, P1, P2

        bboxes, scores = model.get_pred(
            outputs=outputs,
            img_shape=img_scale,
            anchors=ANCHORS,
            anchor_masks=ANCHOR_MASKS,
            valid_thresh=VALID_THRESH
        )
        # boxes.shape: [N, 框的数量, 4]
        # scores.shape: [N, 7, 框的数量]

        boxes_data = bboxes.numpy()
        scores_data = scores.numpy()
        results = multiclass_nms(
            bboxes=boxes_data,
            scores=scores_data,
            score_thresh=VALID_THRESH,
            nms_thresh=NMS_THRESH,
            pre_nms_topk=NMS_TOPK,
            pos_nms_topk=NMS_POSK
        )
        # results 是有 N 个 list 元素的 list

        result = results[0]
        draw_results(result=result, file_path=image_path, draw_thresh=0.5)

    plt.show()
