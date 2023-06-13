import time
import numpy as np
import paddle

from datasets.InsectsDataset import TrainDataset, test_data_loader
from net.YOLOv3 import YOLOv3

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
IGNORE_THRESH = 0.7
NUM_CLASSES = 7


def get_lr(base_lr=0.0001, lr_decay=0.1):
    # 学习率变化的边界步数
    bd = [3000, 13000]
    # 学习率变化的值
    lr = [base_lr, base_lr * lr_decay,
          base_lr * lr_decay * lr_decay]  # 在第[0,10000), [10000,20000), [20000,+∞)分别对应 value 中学习率的值
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)
    return learning_rate


if __name__ == '__main__':

    paddle.device.set_device('gpu:0')

    TRAINDIR = 'data/insects/train'
    VALIDDIR = 'data/insects/val'
    # 实例化数据集类
    train_dataset = TrainDataset(TRAINDIR, mode='train')
    valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 创建数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=10, shuffle=True,
                                        drop_last=True, use_shared_memory=False)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=10, shuffle=True,
                                        drop_last=True, use_shared_memory=False)

    # 创建模型
    model = YOLOv3(num_classes=NUM_CLASSES)
    # 加载模型参数
    state_dict = paddle.load('checkpoint/yolo_epoch40')
    model.load_dict(state_dict)

    learning_rate = get_lr()
    opt = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=paddle.regularizer.L2Decay(0.0005),
        parameters=model.parameters()
    )

    MAX_EPOCH = 200
    for epoch in range(41, MAX_EPOCH):
        # -----------------------------------------训练-----------------------------------------
        model.train()
        for iter, data in enumerate(train_loader()):
            # 清除梯度
            opt.clear_grad()

            # img: [N, C=3, H, W], gt_boxes: [N, 50, 4], gt_labels: [N, 50], img_scale: [N, 2] (原始图片的高宽)
            img, gt_boxes, gt_labels, img_scale = data
            # 创建真实框置信度
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)

            # 前向传播，输出 [P0, P1, P2]
            # P0: [N, num_anchors * (5 + num_classes), 20, 20]
            # P1: [N, num_anchors * (5 + num_classes), 40, 40]
            # P2: [N, num_anchors * (5 + num_classes), 80, 80]
            outputs = model(img)

            loss = model.get_loss(outputs, gt_boxes, gt_labels,
                                  gt_score=gt_scores,
                                  anchors=ANCHORS,
                                  anchor_mask=ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)  # shape: [1, ]

            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            opt.step()

            if iter % 10 == 0:
                time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print(f'{time_string}[TRAIN]epoch {epoch}, iter {iter}, output loss: {loss.numpy()}')

        # 保存模型参数
        if epoch % 10 == 0 or epoch == MAX_EPOCH - 1:
            paddle.save(model.state_dict(), f'checkpoint/yolo_epoch{epoch}')
            print('Saved model!')

        # -----------------------------------------验证-----------------------------------------
        # 每个 epoch 结束之后，在验证集上进行测试
        model.eval()
        for iter, data in enumerate(valid_loader()):
            # img: [N, C=3, H, W], gt_boxes: [N, 50, 4], gt_labels: [N, 50], img_scale: [N, 2] (原始图片的高宽)
            img, gt_boxes, gt_labels, img_scale = data
            # 创建真实框置信度
            gt_scores = np.ones(gt_labels.shape).astype('float32')
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)

            # 前向传播，输出 [P0, P1, P2]
            # P0: [N, num_anchors * (5 + num_classes), 20, 20]
            # P1: [N, num_anchors * (5 + num_classes), 40, 40]
            # P2: [N, num_anchors * (5 + num_classes), 80, 80]
            outputs = model(img)

            loss = model.get_loss(outputs, gt_boxes, gt_labels,
                                  gt_score=gt_scores,
                                  anchors=ANCHORS,
                                  anchor_mask=ANCHOR_MASKS,
                                  ignore_thresh=IGNORE_THRESH,
                                  use_label_smooth=False)  # shape: [1, ]
            if iter % 1 == 0:
                time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print(f'{time_string}[VALID]epoch {epoch}, iter {iter}, output loss: {loss.numpy()}')
