import numpy as np
import matplotlib.patches as patches


def draw_rectangle(currentAxis, bbox, edgecolor = 'k', facecolor = 'y', fill=False, linestyle='-'):
    """
    画矩形框的函数
    :param currentAxis: 坐标轴，通过 plt.gca() 获取
    :param bbox: 边界框，包含四个数值的 list，[x1, y1, x2, y2]
    :param edgecolor: 边框线条颜色
    :param facecolor: 填充颜色
    :param fill: 是否填充
    :param linestyle: 边框线型
    :return:
    """
    # patches.Rectangle 需要传入左上角坐标，矩形区域的宽度、高度等参数
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1,
                            edgecolor=edgecolor, facecolor=facecolor, fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)


def box_iou_xyxy(box1, box2):
    """
    计算两个矩形框的 IoU，矩形框的格式是 xyxy
    :param box1: 矩形框 1
    :param box2: 矩形框 2
    :return:
    """
    # tensor 转 ndarray
    # if type(box1) is paddle.Tensor:
    #     box1 = box1.numpy()
    # if type(box2) is paddle.Tensor:
    #     box2 = box2.numpy()

    # 获取 box1 左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算 box1 的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取 box2 左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算 box2 的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形框的高度、宽度和面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并的面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou


def box_iou_xywh(box1, box2):
    """
    计算两个矩形框的 IoU，矩形框的格式是 xywh
    :param box1: 矩形框 1
    :param box2: 矩形框 2
    :return:
    """
    x1min, y1min = box1[0] - box1[2] / 2., box1[1] - box1[3] / 2.
    x1max, y1max = box1[0] + box1[2] / 2., box1[1] + box1[3] / 2.
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2] / 2., box2[1] - box2[3] / 2.
    x2max, y2max = box2[0] + box2[2] / 2., box2[1] + box2[3] / 2.
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


def multi_box_iou_xywh(box1, box2):
    """
    计算两组矩形框的 IoU
    box1 或 box2 可以包含多个盒子。
    此方法只能处理两种情况：
        1、box1 和 box2 具有相同的形状，即 box1.shape == box2.shape
        2、box1 或 box2 只包含一个盒子，即 len(box1) == 1 或 len(box2) == 1
    如果 box1 和 box2 的形状不匹配，并且它们都包含多个盒子，那么会报错。
    """
    assert box1.shape[-1] == 4, 'Box1 shape[-1] should be 4.'
    assert box2.shape[-1] == 4, 'Box2 shape[-1] should be 4.'

    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2., box1[:, 1] - box1[:, 3] / 2.
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2., box1[:, 1] + box1[:, 3] / 2.
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 0] / 2., box2[:, 1] - box2[:, 3] / 2.
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 0] / 2., box2[:, 1] + box2[:, 3] / 2.

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0, a_max=None)
    inter_h = np.clip(inter_h, a_min=0, a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    """
    裁剪框
    boxes: xywh 形式, shape: [N, 4]
    labels: shape: [N, ]
    crop: 需要裁剪的区域, xywh 形式(这里的 xy 是裁剪区域的左上角的坐标)
    img_shape: 格式: (w, h)
    """
    x, y, w, h = map(float, crop)
    img_w, img_h = map(float, img_shape)

    # 将边界框 转换为 xyxy 形式，并从归一化坐标转换为像素坐标
    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2.) * img_w, \
                               (boxes[:, 0] + boxes[:, 2] / 2.) * img_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2.) * img_h, \
                               (boxes[:, 1] + boxes[:, 3] / 2.) * img_h

    crop_box = np.array([x, y, x + w, y + h])  # 裁剪框，xyxy 形式
    # 计算所有边界框的中心点坐标
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # 即 (x1 + x2) / 2. 和 (y1 + y2) / 2.
    # 求所有边界框的中心点 是否 在裁剪框内，结果为(N, 2) 的布尔型数组
    # all(axis=1) 会将结果压缩成一个形如 (N,) 的布尔型数组
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    # boxes 的左上角或右下角坐标，超出裁剪框的，将其缩回到裁剪框的左上角或右下角
    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    # 将边界框的坐标 转换为 相对于裁剪框的坐标
    # 因为裁剪图像后，原点就变成了裁剪框的左上角，所以需要将边界框的坐标转换成相对于裁剪框的坐标
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    # boxes[:, :2] < boxes[:, 2:] 以筛选出有效的边界框
    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1)).astype('float32')
    # 将无效的边界框的坐标和大小都设置为0，从而将边界框限制在掩码内部
    boxes = boxes * np.expand_dims(mask, axis=1)

    labels = labels * mask
    # 将边界框 转换为 xywh 形式，并从像素坐标转换为归一化坐标（相对于裁剪框宽高的比例）
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2. / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2. / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


def nms(bboxes, scores, score_thresh, nms_thresh, pre_nms_topk, i=0, c=0):
    """
    非极大值抑制
    :param bboxes: 预测框, shape: [框的数量, 4]
    :param scores: 预测框对应的得分, shape: [框的数量, ]
    :param score_thresh: 得分阈值
    :param nms_thresh: 非极大值抑制阈值
    :param pre_nms_topk:
    :param i:
    :param c:
    :return: 保留下来的预测框的下标
    """
    remain_inds = np.argsort(scores)    # 将得分按 从小到大 排序，并返回对应的下标
    remain_inds = remain_inds[::-1]     # 将下标翻转（相当于得分从大到小排序）
    keep_inds = []                      # 保留的预测框的下标

    while len(remain_inds) > 0:
        cur_ind = remain_inds[0]        # 取出得分最大的预测框的下标
        cur_score = scores[cur_ind]     # 取出得分最大的预测框的得分
        # 如果 预测框得分 < 得分阈值，就结束了，因为排过序了，后面的预测框的得分也是小于得分阈值的
        if cur_score < score_thresh:
            break

        keep = True
        # 待判断的预测框
        current_box = bboxes[cur_ind]
        for ind in keep_inds:
            # 已经保留的预测框
            keep_box = bboxes[ind]
            # 计算 待判断的预测框 与 已经保留的预测框 的 IoU
            iou = box_iou_xyxy(current_box, keep_box)
            # 如果 IoU > nms阈值，则将该预测框丢弃
            if iou > nms_thresh:
                keep = False
                break

        if i == 0 and c == 4 and cur_ind == 951:
            print('suppressed, ', keep, i, c, cur_ind, ind, iou)
        if keep:
            keep_inds.append(cur_ind)
        remain_inds = remain_inds[1:]
    return np.array(keep_inds)


def multiclass_nms(bboxes, scores, score_thresh=0.01, nms_thresh=0.45, pre_nms_topk=1000, pos_nms_topk=100):
    """
    多分类非极大值抑制
    :param bboxes: 预测框, shape: [N, 框的数量, 4]
    :param scores: 预测框对应的得分, shape: [N, 7, 框的数量]
    :param score_thresh: 得分阈值
    :param nms_thresh: 非极大值抑制阈值
    :param pre_nms_topk:
    :param pos_nms_topk:
    :return:
    """
    batch_size = bboxes.shape[0]
    class_num = scores.shape[1]
    rets = []

    for i in range(batch_size):
        bboxes_i = bboxes[i]    # shape: [框的数量, 4]
        scores_i = scores[i]    # shape: [7, 框的数量]
        ret = []

        for c in range(class_num):
            scores_i_c = scores_i[c]    # 第 i 张图像上的每个预测框属于 c 类的得分（置信度）, shape: [框的数量]
            keep_inds = nms(bboxes_i, scores_i_c, score_thresh, nms_thresh, pre_nms_topk, i=i, c=c)
            # 如果该类没有保留的预测框，则下一个类
            if len(keep_inds) < 1:
                continue

            keep_bboxes = bboxes_i[keep_inds]
            # 如果 len(keep_inds) = 1, 则 keep_bboxes 的形状是 [4]
            # 则还需要对 keep_bboxes 进行扩充维度, 转变成 [1, 4]
            if len(keep_inds) == 1:
                keep_bboxes = np.reshape(keep_bboxes, [1, -1])
            keep_scores = scores_i_c[keep_inds]

            keep_results = np.zeros([keep_scores.shape[0], 6])
            keep_results[:, 0] = c
            keep_results[:, 1] = keep_scores[:]
            keep_results[:, 2:6] = keep_bboxes[:, :]

            ret.append(keep_results)
        if len(ret) < 1:
            rets.append(ret)
            continue

        ret_i = np.concatenate(ret, axis=0)
        scores_i = ret_i[:, 1]
        if len(scores_i) > pos_nms_topk:
            inds = np.argsort(scores_i)[::-1]
            inds = inds[:pos_nms_topk]
            ret_i = ret_i[inds]

        rets.append(ret_i)

    return rets
