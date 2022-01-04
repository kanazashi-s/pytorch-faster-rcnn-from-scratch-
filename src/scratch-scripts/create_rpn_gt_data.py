#%%

# アンカーボックスのパターンは全部で9パターン
# アンカーボックスの表現はx1, y1, x2, y2で行われるから、アンカーボックスのNumpPy配列のshapeは(9, 4)

import numpy as np
anchor_ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

anchor_base = np.zeros((len(anchor_ratios) * len(anchor_scales), 4), dtype=np.float32)

#%%

sub_sample = 16 # 特徴マップ上のサイズ × sub_sample = 元画像上のサイズ

center_y = sub_sample / 2.
center_x = sub_sample / 2.

for i, ratio in enumerate(anchor_ratios):
    for j, scale in enumerate(anchor_scales):
        h = sub_sample * scale * np.sqrt(ratio)
        w = sub_sample * scale * np.sqrt(1./ratio)

        index = i * 3 + j

        anchor_base[index, 0] = center_y - h / 2.
        anchor_base[index, 1] = center_x - w / 2.
        anchor_base[index, 2] = center_y + h / 2.
        anchor_base[index, 3] = center_x + w / 2.

# anchor_base is the anchor boxes at the first pixel of feature map.
# Let's generate all anchors : 9 * (50 * 50) pixels.
# At first, calculate all the centers of image at every feature map pixels.

#%%

import itertools

feature_map_size = 800 // sub_sample
centers_x = np.arange(sub_sample, (feature_map_size+1) * sub_sample, sub_sample)
centers_y = np.arange(sub_sample, (feature_map_size+1) * sub_sample, sub_sample)

index = 0
centers = np.zeros(shape=(len(centers_x) * len(centers_y), 2))
for center_x, center_y in itertools.product(centers_x, centers_y):
    centers[index, 0] = center_x
    centers[index, 1] = center_y
    index += 1

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
plt.scatter(x=centers[:, 0], y=centers[:, 1])
plt.show()

#%%

anchors = np.zeros(shape=(feature_map_size*feature_map_size*9, 4))
for i, center in enumerate(centers):
    for j, (ratio, scale) in enumerate(itertools.product(anchor_ratios, anchor_scales)):

        anchor_height = sub_sample * scale * np.sqrt(ratio)
        anchor_width = sub_sample * scale * np.sqrt(1/ratio)

        anchors[i*9+j, 0] = center[0] - anchor_height / 2
        anchors[i*9+j, 1] = center[1] - anchor_width / 2
        anchors[i*9+j, 2] = center[0] + anchor_height / 2
        anchors[i*9+j, 3] = center[1] + anchor_width / 2

#%%

# 2つのアンカーボックスに Positive label を与えてみう

# Guidelines of using bounding boxes in faster-rcnn detection flow.
# a) GT の BBox と最も大きいIoUスコアを出したアンカーボックス -> Positive
# b) or GT の BBox とのIoUスコアが0.7以上のアンカーボックス -> Positive
# ↑Note that たくさんのアンカーボックスが単一の GT の BBox に
# c) 全ての GT アンカーボックスとの IoU スコアが0.3以下のアンカーボックス -> Negative
# d) Positive でも Negative でもないアンカーボックスは訓練に影響を与えない

bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
labels = np.asarray([6, 8], dtype=np.int8) # 0 represents background

#%%

anchors_inside_img = np.where(
    (anchors[:, 0] >= 0) &
    (anchors[:, 1] >= 1) &
    (anchors[:, 2] <= 800) &
    (anchors[:, 3] <= 800)
)

print(anchors_inside_img[0])

label = np.empty(shape=(len(anchors_inside_img[0]), ), dtype=np.int32)
label.fill(-1)
valid_anchors = anchors[anchors_inside_img]

#%%

def calc_iou(bbox_a, bbox_b):
    ya1, xa1, ya2, xa2 = bbox_a
    bbox_a_area = (ya2 - ya1) * (xa2 - xa1)

    yb1, xb1, yb2, xb2 = bbox_b
    bbox_b_area = (yb2 - yb1) * (xb2 - xb1)

    inter_x1 = max([xb1, xa1])
    inter_y1 = max([yb1, ya1])
    inter_x2 = min([xb2, xa2])
    inter_y2 = min([yb2, ya2])

    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
        inter_area = (inter_y2 - inter_y1) * \
                    (inter_x2 - inter_x1)
        iou = inter_area / (bbox_a_area + bbox_b_area - inter_area)
    else:
        iou = 0.

    return iou

ious = np.zeros((len(valid_anchors), 2), dtype=np.float32)
for i, anchor in enumerate(valid_anchors):
    for j, gt_bbox in enumerate(bbox):
        ious[i, j] = calc_iou(anchor, gt_bbox)

#%%

# それぞれのGT BBox に対する最大のIoU値と対応するアンカーボックスはどれ？
anchors_max_ious_each_gts = ious.argmax(axis=0)
max_ious_for_gts = ious[anchors_max_ious_each_gts, np.arange(ious.shape[1])]

# それぞれのアンカーボックスに対する最大のIoU値と対応する GT BBox はどれ？
gts_max_ious_each_anchors = ious.argmax(axis=1)
max_ious_for_anchor = ious[np.arange(len(ious)), gts_max_ious_each_anchors]

pos_iou_threshold  = 0.7
neg_iou_threshold = 0.3

#%%

# c) 全ての GT アンカーボックスとの IoU スコアが0.3以下のアンカーボックス -> Negative
label[max_ious_for_anchor < neg_iou_threshold] = 0

# a) GT の BBox と最も大きいIoUスコアを出したアンカーボックス -> Positive
label[anchors_max_ious_each_gts] = 1

# b) GT の BBox とのIoUスコアが0.7以上のアンカーボックス -> Positive
label[max_ious_for_anchor > pos_iou_threshold] = 1

#%%

pos_ratio = 0.5
n_sample = 256
n_pos = pos_ratio * n_sample

# pos_index が n_pos よりも多かった場合、余ったラベルをランダムに -1 に変える
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(
        pos_index,
        size=(len(pos_index) - n_pos),
        replace=False
    )
    label[disable_index] = -1

n_neg = n_sample * np.sum(label == 1)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
    disable_index = np.random.choice(
        neg_index,
        size=(len(neg_index) - n_neg),
        replace = False
    )
    label[disable_index] = -1

#%%

def convert_coordinates_to_centers(bbox_coordinate_format):
    """convert bbox information from
    (y1, x1, y2, x2) format to
    (center_y, center_x, height, width) format."""

    height = bbox_coordinate_format[:, 2] - bbox_coordinate_format[:, 0]
    width = bbox_coordinate_format[:, 3] - bbox_coordinate_format[:, 1]
    center_y = bbox_coordinate_format[:, 0] + 0.5 * height
    center_x = bbox_coordinate_format[:, 1] + 0.5 * width

    return center_y, center_x, height, width

#%%

# Anchor Boxes に location をアサインする
# GT Box のロケーション (正解データ) を各 Anchor Box に割り当てるイメージか
# それぞれの Anchor Box に最も近い GT Box の座標が割り当てられるイメージっぽい？

max_iou_bbox_each_anchors = bbox[gts_max_ious_each_anchors]
gt_center_y, gt_center_x, gt_height, gt_width = convert_coordinates_to_centers(max_iou_bbox_each_anchors)

#%%

eps = 1e-3

center_y, center_x, height, width = convert_coordinates_to_centers(valid_anchors)

height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (gt_center_y - center_y) / height
dx = (gt_center_x - center_x) / width
dh = np.log(gt_height / height)
dw = np.log(gt_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

#%%

# final anchor_labels
anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[anchors_inside_img] = label

#%%

# final anchor_locations
anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[anchors_inside_img, :] = anchor_locs

# The final two matrices.
# 1. anchor_labels : (22500, ) : -1ignore, 0background, 1object
# 2. anchor_locations : (22500, 4) : differences between anchors and gt_bboxes.