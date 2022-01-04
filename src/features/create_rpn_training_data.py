import itertools
from typing import Tuple

import numpy as np

def main():
    centers = create_all_anchor_centers()
    anchors = create_anchors_for_all_fmap_pixels()
    labels = create_labels_for_all_anchors()
    labels = suppress_num_labels()
    assign_locs_to_anchors(anchors)


def create_all_anchor_centers(
        img_size: Tuple =(3, 800, 800),
        sub_sample: int =16,
) -> np.ndarray:

    fmap_height = img_size[1] // sub_sample
    fmap_width = img_size[2] // sub_sample
    centers_x = np.arange(
        sub_sample,
        (fmap_width + 1) * sub_sample,
        sub_sample
    )
    centers_y = np.arange(
        sub_sample,
        (fmap_height + 1) * sub_sample,
        sub_sample
    )

    index = 0
    centers = np.zeros(shape=(len(centers_x) * len(centers_y), 2))
    for center_x, center_y in itertools.product(centers_x, centers_y):
        centers[index, 0] = center_x
        centers[index, 1] = center_y
        index += 1

    return centers

def create_anchors_for_all_fmap_pixels(
        img_size: Tuple =(3, 800, 800),
        sub_sample: int =16,
        anchor_ratios: Tuple = (0.5, 1, 2),
        anchor_scales: Tuple = (8, 16, 32),
):
    fmap_height = img_size[1] // sub_sample
    fmap_width = img_size[2] // sub_sample

    anchors = np.zeros(shape=(fmap_height * fmap_width * 9, 4))
    for i, center in enumerate(centers):
        for j, (ratio, scale) in enumerate(itertools.product(anchor_ratios, anchor_scales)):
            anchor_height = sub_sample * scale * np.sqrt(ratio)
            anchor_width = sub_sample * scale * np.sqrt(1 / ratio)

            anchors[i * 9 + j, 0] = center[0] - anchor_height / 2
            anchors[i * 9 + j, 1] = center[1] - anchor_width / 2
            anchors[i * 9 + j, 2] = center[0] + anchor_height / 2
            anchors[i * 9 + j, 3] = center[1] + anchor_width / 2

    return anchors

def create_labels_for_all_anchors(
        gt_bbox=np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32), # [y1, x1, y2, x2] format
        gt_labels=np.asarray([6, 8], dtype=np.int8),
        pos_iou_threshold=0.7,
        neg_iou_threshold=0.3,
):

    valid_anchor_indices = create_valid_anchor_indices(anchors)
    valid_anchors = anchors[valid_anchor_indices]

    labels = np.empty(shape=(len(valid_anchors[0]),), dtype=np.int32)
    labels.fill(-1)

    ious = create_iou_matrices(valid_anchors, gt_bbox)

    anchor_idx_max_ious_each_gts = ious.argmax(axis=0)
    max_iou_vals_each_anchor = ious[anchor_idx_max_ious_each_gts, np.arange(ious.shape[1])]
    gt_idx_max_ious_each_anchors = ious.argmax(axis=1)

    # 全ての GT アンカーボックスとの IoU スコアが0.3以下のアンカーボックス -> Negative
    labels[max_iou_vals_each_anchor < neg_iou_threshold] = 0

    # GT の BBox と最も大きいIoUスコアを出したアンカーボックス -> Positive
    labels[anchor_idx_max_ious_each_gts] = 1

    # GT の BBox とのIoUスコアが0.7以上のアンカーボックス -> Positive
    labels[max_iou_vals_each_anchor > pos_iou_threshold] = 1

    anchor_labels = np.empty((len(anchors),), dtype=labels.dtype)
    anchor_labels.fill(-1)
    anchor_labels[valid_anchor_indices] = labels

    return labels

def create_valid_anchor_indices(
        anchors :np.ndarray,
        img_size :Tuple =(3, 800, 800)
):
    valid_anchor_indices = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= img_size[1]) &
        (anchors[:, 3] <= img_size[2])
    )

    return valid_anchor_indices

def calc_iou(
        bbox_a,
        bbox_b
):
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

def create_iou_matrices(
        anchors: np.ndarray,
        gt_bbox: np.ndarray,
):
    ious = np.zeros((len(anchors), 2), dtype=np.float32)
    for i, anchor in enumerate(anchors):
        for j, gt_bbox in enumerate(gt_bbox):
            ious[i, j] = calc_iou(anchor, gt_bbox)

    return ious

def suppress_num_labels(
        labels,
        batch_size: int = 256,
        pos_ratio: float = 0.5,
):

    n_pos = pos_ratio * batch_size
    # pos_index が n_pos よりも多かった場合、余ったラベルをランダムに -1 に変える
    pos_index = np.where(labels == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(
            pos_index,
            size=(len(pos_index) - n_pos),
            replace=False
        )
        labels[disable_index] = -1

    n_neg = (1-pos_ratio) * batch_size
    neg_index = np.where(labels == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(
            neg_index,
            size=(len(neg_index) - n_neg),
            replace=False
        )
        labels[disable_index] = -1

def assign_locs_to_anchors(
        anchors,
        gt_bbox=np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32), # [y1, x1, y2, x2] format
):
    valid_anchor_indices = create_valid_anchor_indices(anchors)
    valid_anchors = anchors[valid_anchor_indices]

    ious = create_iou_matrices(anchors, gt_bbox)
    gts_max_ious_each_anchors = ious.argmax(axis=1)

    max_iou_bbox_each_anchors = gt_bbox[gts_max_ious_each_anchors]
    gt_center_y, gt_center_x, gt_height, gt_width = convert_coordinates_to_centers(max_iou_bbox_each_anchors)

    eps = 1e-5
    center_y, center_x, height, width = convert_coordinates_to_centers(valid_anchors)

    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)
    anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()

    anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[valid_anchor_indices, :] = anchor_locs

    return anchor_locs

def convert_coordinates_to_centers(bbox_coordinate_format):
    """convert bbox information from
    (y1, x1, y2, x2) format to
    (center_y, center_x, height, width) format."""

    height = bbox_coordinate_format[:, 2] - bbox_coordinate_format[:, 0]
    width = bbox_coordinate_format[:, 3] - bbox_coordinate_format[:, 1]
    center_y = bbox_coordinate_format[:, 0] + 0.5 * height
    center_x = bbox_coordinate_format[:, 1] + 0.5 * width

    return center_y, center_x, height, width

if __name__ == '__main__':
    centers = create_all_anchor_centers()
    anchors = create_anchors_for_all_fmap_pixels()
    labels = create_labels_for_all_anchors()
    labels = suppress_num_labels()
    assign_locs_to_anchors(anchors)

