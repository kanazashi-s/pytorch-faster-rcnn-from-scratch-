# 画像・バウンディングボックス・ラベルのセットを準備する

from typing import Tuple
from collections import deque

import torch
import torch.nn as nn
import torchvision

SUB_SAMPLE = 16

def get_model(
        img_size: Tuple =(3, 800, 800)):
    img_size = deque(img_size)
    img_size.appendleft(1) # add batch_size dimension.

    dummy_img = torch.zeros(tuple(img_size)).float()
    model = torchvision.models.vgg16(pretrained=False)
    vgg_layers = list(model.features)

    req_features = []
    k = dummy_img.clone()
    for i in vgg_layers:
        k = i(k)
        if k.size()[2] < 800 // 16:
            break
        req_features.append(i)
        out_channels = k.size()[1]

    # 特徴量抽出器の完成
    faster_rcnn_feature_extractor = nn.Sequential(*req_features)
    return faster_rcnn_feature_extractor, out_channels

if __name__ == '__main__':
    model, out_channels = get_model()