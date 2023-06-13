import json
import paddle

from datasets.InsectsDataset import test_data_loader
from net.YOLOv3 import YOLOv3
from utils.box_utils import multiclass_nms

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHORS_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
VALID_THRESH = 0.01
NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45
NUM_CLASSES = 7

if __name__ == '__main__':

    # paddle.device.set_device('gpu:0')
    paddle.device.set_device('cpu')

    TESTDIR = 'data/insects/test/images'

    # 创建数据读取器
    test_loader = test_data_loader(datadir=TESTDIR, batch_size=10)

    # 创建模型
    model = YOLOv3(num_classes=NUM_CLASSES)
    params_file_path = 'checkpoint/yolo_epoch40'
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    total_results = []
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale = data
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale)

        outputs = model(img)

        bboxes, scores = model.get_pred(
            outputs=outputs,
            img_shape=img_scale,
            anchors=ANCHORS,
            anchor_masks=ANCHORS_MASK,
            valid_thresh=VALID_THRESH
        )
        # boxes.shape: [N, 框的数量, 4]
        # scores.shape: [N, 7, 框的数量]

        bboxes_data = bboxes.numpy()
        scores_data = scores.numpy()
        results = multiclass_nms(
            bboxes=bboxes_data,
            scores=scores_data,
            score_thresh=VALID_THRESH,
            nms_thresh=NMS_THRESH,
            pre_nms_topk=NMS_TOPK,
            pos_nms_topk=NMS_POSK
        )
        # results 是有 N 个 list 元素的 list

        for j in range(len(results)):
            result_j = results[j]
            img_name_j = img_name[j]
            total_results.append([img_name_j, result_j])
        print(f'processed {len(total_results)} pictures')

    print('')
    json.dump(total_results, open('pred_results.json', 'w'))
