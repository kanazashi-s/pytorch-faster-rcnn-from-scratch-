#%%

# 画像・バウンディングボックス・ラベルのセットを準備する

import torch
import torch.nn as nn
image = torch.zeros((1, 3, 800, 800)).float()

bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
labels = torch.LongTensor([6, 8]) # 0 represents background
sub_sample = 16

#%%

# VGG16を、バックボーンに使用する
# VGG16の出力特徴マップのサイズが 800//16 = 50 になるよう、小細工をする

import torchvision

dummy_img = torch.zeros((1, 3, 800, 800)).float()
model = torchvision.models.vgg16(pretrained=False)
vgg_layers = list(model.features)

req_features = []
k = dummy_img.clone()
for i in vgg_layers:
    k = i(k)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]

# 特徴量抽出器の完成
faster_rcnn_fe_extractor = nn.Sequential(*req_features)