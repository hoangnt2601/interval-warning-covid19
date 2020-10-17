"""
Chọn toạ độ 4 điểm sao cho từ hướng camera loe ra
"""

import cv2
import numpy as np
from camera.video import FileVideoStream, VideoStream
from social_distances import dist, plot, calib
import json


mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append([x, y])
    print("mouse_pts ", mouse_pts)


bird_pts = []


def get_bird_points(event, x, y, flags, param):
    global bird_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if "bird_pts" not in globals():
            bird_pts = []
        bird_pts.append([x, y])
    print("bird_pts ", bird_pts)


if __name__ == "__main__":
    vs = FileVideoStream("../video/pedestrians.mp4").start()

    cv2.namedWindow("a")
    cv2.setMouseCallback("a", get_mouse_points)

    cv2.namedWindow("b")
    cv2.setMouseCallback("b", get_bird_points)

    coords = {}
    global image

    perspective_data = {}

    while True:
        frame = vs.read()
        if frame is None:
            break

        image = frame.copy()

        cv2.imshow("f", image)
        key = cv2.waitKey(20)

        if key == ord("q"):
            break
        if key == ord("d"):
            cv2.imshow("a", image)
            key = cv2.waitKey(0)
        if key == ord("c"):
            offsets = np.array(mouse_pts[:4], dtype=np.float32)
            perspective_image, perspective_matrix = calib.get_perspective_transform(
                image, offsets
            )
            cv2.imshow("b", perspective_image)
            cv2.waitKey(0)
        coords["offsets"] = mouse_pts
        coords["scales"] = bird_pts
        with open("coords.json", "w") as outfile:
            json.dump(coords, outfile)
    vs.stop()
    cv2.destroyAllWindows()
