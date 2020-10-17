import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from numba import jit
except:

    def jit(func):
        return func


np.random.seed(0)


def linear_assignment(cost_matrix):
    """Hungarian: Thuật toán phân công sao cho chi phí là tối ưu nhất
    Params:
            - cost_matrix: Ma trận chi phí (vị trí) của det và trk
    Return:
            Các cặp vị trí của det và trk với chi phí ít nhất và khớp nhau nhất"""

    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


# TODO: batch iou
@jit
def iou(det, trk):
    """
    Tính toán IOU giữa bbox và tracker [x1,y1,x2,y2]
    """
    """Chọn x1, y1 lớn nhất giữa det và trk, x2, y2 nhỏ nhất giữa det và trk
	để luôn chọn được bb nằm gọn và fit nhất trong object giữa det và trk"""
    xx1 = np.maximum(det[0], trk[0])
    yy1 = np.maximum(det[1], trk[1])
    xx2 = np.minimum(det[2], trk[2])
    yy2 = np.minimum(det[3], trk[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    # Tỉ lệ w,h giao nhau
    o = wh / (
        (det[2] - det[0]) * (det[3] - det[1])
        + (trk[2] - trk[0]) * (trk[3] - trk[1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Lấy một bbox ở dạng [x1, y1, x2, y2] và trả về z ở dạng [x, y, s, r]
    trong đó x, y là tâm của bbx và s là tỷ lệ (scale) và r là tỷ lệ khung hình (ratio)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Lấy một bbox ở dạng trung tâm [x, y, s, r] và trả về nó ở dạng [x1, y1, x2, y2]
    trong đó x1, y1 là trên cùng bên trái và x2, y2 là dưới cùng bên phải
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Trả về danh sách sự liên kết các matched, unatched_detections và unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)  # vị trị của cặp det - trk

    # Sử dụng thuật toán Hungarian để lọc
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # lọc các liên kết với IOU thấp
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
