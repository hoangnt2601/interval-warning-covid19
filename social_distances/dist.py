"""
Contains functions to calculate bottom center for all bounding boxes and transform prespective for all points,
calculate distance between humans, calculate width and height scale ratio for bird eye view,
and calculates number of humans at risk, low risk, no risk according to closeness.
"""

# imports
import cv2
import numpy as np
import math
from itertools import product
import collections as colls


def distance_points(p1, p2, perspective_data):
    """Khoảng tỉ lệ theo vertical và horizontal từ bird-eye so với real world"""
    scale_w, scale_h = perspective_data["scale_wh"]
    dx = abs(p2[0] - p1[0]) * scale_w
    dy = abs(p2[1] - p1[1]) * scale_h
    d = abs(math.sqrt(dx ** 2 + dy ** 2))

    return round(d, 2)


def get_distances(centroids, birdeye_points, perspective_data, threshold=2):
    """
    Hàm tính toán khoảng cách giữa tất cả các cặp điểm và tính toán tỷ lệ những obj gần nhau.
    Params:
            centroids: Toạ độ tâm các obj
            birdeye_points: Toạ độ trung điểm phía dưới của obj
            scale_w, scale_h: Tỉ lệ chiều dài, chiều rộng tham chiếu sang bird-eye
            threshold: Ngưỡng cảnh báo, mặc định: 2 mét
    Return:
            Trả về ma trận khoảng cách theo bird-eye và tập tâm và trạng thái theo góc camera
    """

    birdeye_centroids = []
    camera_centroids = []

    list_indexes = np.arange(len(birdeye_points)).tolist()

    for i, j in product(list_indexes, list_indexes):
        perA = birdeye_points[i]
        perB = birdeye_points[j]
        dist = distance_points(perA, perB, perspective_data)
        if i != j:
            if 0 < dist < threshold:
                # Khoảng cách nguy hiểm
                warning = True
                birdeye_centroids.append([perA, perB, warning])
                camera_centroids.append([centroids[i], centroids[j], warning])
            else:
                # Khoảng cách an toàn
                warning = False
                birdeye_centroids.append([perA, perB, warning])
                camera_centroids.append([centroids[i], centroids[j], warning])

    return birdeye_centroids, camera_centroids


def get_scale(W, H):
    """
    Tỉ lệ khung hình birdeye
    """
    scale_w = 480
    scale_h = 720

    return float(scale_w / W), float(scale_h / H)
