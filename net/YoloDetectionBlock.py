import numpy as np
import paddle
from net.Darknet import ConvBNLayer, darknet53


class YoloDetectionBlock(paddle.nn.Layer):
    """
    定义 YOLOv3 检测头。
    使用多层卷积 和 BN 提取特征。
    """
    def __init__(self, ch_in, ch_out, is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0, f'channel {ch_out} cannot be divided by 2.'

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = ConvBNLayer(
            ch_in=ch_out*2,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.route = ConvBNLayer(
            ch_in=ch_out*2,
            ch_out=ch_out,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.tip = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out*2,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


if __name__ == '__main__':

    paddle.device.set_device('gpu:0')

    NUM_ANCHORS = 3
    NUM_CLASSES = 7
    num_filters = NUM_ANCHORS * (5 + NUM_CLASSES)

    backbone = darknet53()
    detection = YoloDetectionBlock(ch_in=1024, ch_out=512)
    conv2d_pred = paddle.nn.Conv2D(
        in_channels=1024,
        out_channels=num_filters,
        kernel_size=1
    )

    x = np.random.randn(1, 3, 640, 640).astype('float32')
    x = paddle.to_tensor(x)
    C0, C1, C2 = backbone(x)
    route, tip = detection(C0)
    P0 = conv2d_pred(tip)

    print(P0.shape)