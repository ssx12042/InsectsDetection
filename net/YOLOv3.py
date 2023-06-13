import paddle
import paddle.nn.functional as F

from net.Darknet import darknet53, ConvBNLayer
from net.YoloDetectionBlock import YoloDetectionBlock


# 定义上采样模块
class Upsample(paddle.nn.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # 获取动态上采样输出的 shape
        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, 'int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # 按实际的 shape 来 resize
        # scale: 输入的高度或宽度的乘数因子乘数因子
        out = F.interpolate(x=inputs, scale_factor=self.scale, mode='NEAREST')
        return out


class YOLOv3(paddle.nn.Layer):
    def __init__(self, num_classes=7):
        super(YOLOv3, self).__init__()

        self.num_classes = num_classes
        # 提取图像特征的骨干网络
        self.block = darknet53()
        self.block_outputs = []  # 从 t_i 生成 p_i 的模块
        self.yolo_blocks = []  # 从 c_i 生成 r_i 和 t_i 的模块
        self.route_block_2 = []
        # 生成 3 个层级的特征图 P0, P1, P2
        for i in range(3):
            # 添加从 c_i 生成 r_i 和 t_i 的模块
            yolo_block = self.add_sublayer(
                f'yolo_detection_block_{i}',
                YoloDetectionBlock(
                    ch_in=512 // (2 ** i) * 2 if i == 0 else 512 // (2 ** i) * 2 + 512 // (2 ** i),
                    ch_out=512 // (2 ** i)
                )
            )
            self.yolo_blocks.append(yolo_block)

            num_filters = 3 * (5 + num_classes)

            # 添加从 t_i 生成 p_i 的模块，这是一个 Conv2D 操作，输出通道数为 3 * (5 + num_classes)
            block_out = self.add_sublayer(
                f'block_out_{i}',
                paddle.nn.Conv2D(
                    in_channels=512 // (2 ** i) * 2,
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Normal(0., 0.02),
                    ),
                    bias_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(0.),
                        regularizer=paddle.regularizer.L2Decay(0.)
                    )
                )
            )
            self.block_outputs.append(block_out)
            if i < 2:
                # 对 r_i 进行卷积
                route = self.add_sublayer(
                    f'route2_{i}',
                    ConvBNLayer(
                        ch_in=512 // (2 ** i),
                        ch_out=256 // (2 ** i),
                        kernel_size=1,
                        stride=1,
                        padding=0
                    )
                )
                self.route_block_2.append(route)
            # 将 r_i 放大以便与 c_{i+1} 保持同样的尺寸
            self.upsample = Upsample()

    def forward(self, inputs):
        outputs = []
        blocks = self.block(inputs)  # C0, C1, C2
        for i, block in enumerate(blocks):
            if i > 0:
                # 将 r_{i-1} 经过卷积和上采样之后得到特征图，与这一级的 c_i 进行拼接
                block = paddle.concat([route, block], axis=1)

            # 从 c_i 生成 r_i 和 t_i
            route, tip = self.yolo_blocks[i](block)
            # 从 t_i 生成 p_i
            block_out = self.block_outputs[i](tip)
            # 将 p_i 放入列表
            outputs.append(block_out)

            if i < 2:
                # 对 r_i 进行卷积调整通道数
                route = self.route_block_2[i](route)
                # 对 r_i 进行上采样，使其尺寸和 c_{i+1} 保持一致
                route = self.upsample(route)

        return outputs

    def get_loss(self, outputs, gt_boxes, gt_labels, gt_score=None,
                 anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_thresh=0.7,
                 use_label_smooth=False):
        """

        :param outputs: P0, P1, P2, 形状是 [N, C, H, W], 其中 C = num_anchors * (5 + num_classes)
        :param gt_boxes: 真实框, 形状是 [N, 50, 4]
        :param gt_labels: 真实框对应的类别, 形状是 [N, 50]
        :param gt_score: 真实框的置信度, 在使用了 mixup 技巧时用到
        :param anchors: 锚框
        :param anchor_mask: 筛选锚框的 mask
        :param ignore_thresh: iou 阈值, 默认是0.7
        :param use_label_smooth: 一种训练技巧, 如不使用, 设置为 False
        :return: loss (一个数)
        """
        self.losses = []
        downsample = 32
        # 遍历三个层级分别求 loss
        for i, out in enumerate(outputs):
            anchor_mask_i = anchor_mask[i]
            loss = paddle.vision.ops.yolo_loss(
                x=out,  # out 是 P0, P1, P2 中的一个
                gt_box=gt_boxes,  # 真实框坐标
                gt_label=gt_labels,  # 真实框类别
                gt_score=gt_score,  # 真实框置信度，使用 mixup 训练技巧时需要，不使用该技巧时直接设置为 1，形状与 gt_labels 相同
                anchors=anchors,  # 锚框尺寸，包含 [w0, h0, w1, h1, ..., w8, h8] 共 9 个锚框的尺寸
                anchor_mask=anchor_mask_i,  # 筛选锚框的 mask，例如 anchor_mask_i = [3, 4, 5]，将 anchors 中第 3、4、5 个锚框挑选出来给该层级使用
                class_num=self.num_classes,  # 类别数目
                ignore_thresh=ignore_thresh,  # 当预测框与真实框的 IoU > ignore_thresh 且不是最大的 IoU，标注 objectness = -1
                downsample_ratio=downsample,  # 特征图片相对于原图缩小的倍数，例如 P0 是 32，P1 是 16，P2 是 8
                use_label_smooth=use_label_smooth  # 使用 label smooth 训练技巧时会用到，这里没用此技巧，直接设置为 False
            )
            self.losses.append(paddle.mean(loss))  # 对该层级的每张图片的 loss 求平均，loss.shape: [10, ]
            downsample = downsample // 2  # 下一级特征图的缩放倍数会减半
        return sum(self.losses)  # 将所有层级的 loss 相加

    def get_pred(self,
                 outputs,
                 img_shape=None,
                 anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 valid_thresh=0.01):
        """
        获取预测框和预测得分
        :param outputs: P0, P1, P2, 形状是 [N, C, H, W], 其中 C = num_anchors * (5 + num_classes)
        :param img_shape: 输入图像的尺寸, 形状是 [N, 2]
        :param anchors: 锚框
        :param anchor_masks: 筛选锚框的 mask
        :param valid_thresh: 置信度阈值
        :return: 预测框, 预测得分
        """
        downsample = 32
        total_boxes = []
        total_scores = []

        # 遍历三个层级分别求 预测框和得分
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])

            boxes, scores = paddle.vision.ops.yolo_box(
                x=out,  # 网络输出特征图，P0或者P1、P2
                img_size=img_shape,  # 输入图片尺寸
                anchors=anchors_this_level,  # 使用到的anchor的尺寸
                class_num=self.num_classes,  # 物体类别数
                conf_thresh=valid_thresh,  # 置信度阈值，得分低于该阈值的预测框位置数值不用计算直接设置为0.0
                downsample_ratio=downsample,  # 特征图的下采样比例，例如P0是32，P1是16，P2是8
                name='yolo_box' + str(i)  # 名字，一般无需设置，默认值为None
            )
            # boxes.shape: [N, 框的数量, 4]
            # scores.shape: [N, 框的数量, 7]
            total_boxes.append(boxes)
            scores = paddle.transpose(scores, perm=[0, 2, 1])  # scores.shape: [N, 7, 框的数量]
            total_scores.append(scores)

            downsample = downsample // 2

        yolo_boxes = paddle.concat(total_boxes, axis=1)
        yolo_scores = paddle.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores
