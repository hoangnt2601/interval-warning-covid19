from collections import OrderedDict

from filterpy.kalman import KalmanFilter

from .utils import *


class TrackObject:
    """
    Sử dụng thuật toán Kalman Filter dự đoán vị trí tiếp theo của object dựa vào vật tốc của object di chuyển
    """

    count = 0

    def __init__(self, bbox):
        """
        Khởi tạo tracker
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # Ma trận chuyển đổi trạng thái
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # Ma trận mô hình quan sát
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        # Ma trận hiệp phương sai nhiễu quan sát
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[
            4:, 4:
        ] *= 1000.0  # cung cấp độ không chắc chắn cao cho vận tốc ban đầu không quan sát được
        self.kf.P *= 10.0
        # Ma trận hiệp phương sai nhiễu hệ thống
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0  # Số lần dự đoán vị trí liên tục của object
        self.id = TrackObject.count
        TrackObject.count += 1
        self.history = []  # Tọa độ bbox lịch sử của tracker
        self.hits = 0
        self.hit_streak = 0  # Số lần object xuất hiện trong frame khi đang theo dấu
        self.age = 0  # Tuổi đời của object

        # self.offsets = offsets # Tập các tạo độ zone
        # self.dwell_time = 0  # Thời gian quan tâm của object tại zone
        # self.is_inzone = False  # Trạng thái object xác định bắt đầu tính dwell time
        # self.zone_id = None # zone id hiện tại của object

    def update(self, bbox):
        """
        Cập nhật vector trạng thái với bbox quan sát.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        x1, y1, x2, y2 = bbox

    def predict(self):
        """
        Dự đoán trạng thái và trả về  bbox dự đoán.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0  # mất dấu
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]  # Tọa độ bbox hiện tại

    def get_state(self):
        """
        Trả về bbox hiện tại
        """
        return convert_x_to_bbox(self.kf.x)

    def centroidInZone(self, cx, cy):
        """
        Kiểm tra tọa độ điểm trung tâm của đối tượng có nằm trong zone hay không?
        Chú ý: Tại một thời điểm thì object chỉ đứng trong 1 zone
        Params:
            - cx, cy: Tọa độ trung tâm của bbox
        Return: (id, state) - id của zone và trạng thái của đối tượng
        """
        for zone_id, offset in self.offsets.items():
            x1, y1, x2, y2 = offset

            if (x1 < cx < x2) and (y1 < cy < y2):
                cenInZone = (zone_id, True)
            else:
                cenInZone = (zone_id, False)
        return cenInZone
