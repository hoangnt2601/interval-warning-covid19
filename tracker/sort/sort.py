import time

import numpy as np

from .kalman_object import TrackObject
from .utils import *


class SortTracker:
    def __init__(self, max_age=1, min_hits=3):
        """
        Params:
            - max_age: Tuổi tối đa của đối tượng đang theo dấu sau khi ra khỏi khung hình, quá tuổi thì xóa đối tượng đó đó
            - min_hits: Số lần theo dấu đối tượng tối thiểu
            - trackers: Danh sách các đối tượng theo dấu
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        # Danh sách các object mất dấu được lưu lại zone id và dwell time trước khi xóa
        self.disappeared_objects = []

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
            - dets: Tọa độ nhận diện của object có dạng [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: Hàm này phải được gọi một lần cho mỗi khung ngay cả khi phát hiện trống.
        Returns: Tọa độ đang theo dấu và id của object.
        """
        self.frame_count += 1
        # Lấy tọa độ bbox đã dự đoán
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                # Nếu tọa độ không tồn tại thì xóa tracker đấy đi
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Kiểm tra liên kết giữa bbox được nhận diện và bbox đang theo dấu
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks
        )

        # cập nhật các trackers đã match với bbox đã phát hiện
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Tạo tracker mới cho những bbox chưa được theo dấu
        for i in unmatched_dets:
            trk = TrackObject(dets[i, :])
            self.trackers.append(trk)

        # Xóa tracker đã mất dấu
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                # self.disappeared_objects.append([trk.zone_id, trk.dwell_time])
                self.trackers.pop(i)

        for i, trk in enumerate(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [i])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret).astype(np.int32)
        return np.empty((0, 5))
