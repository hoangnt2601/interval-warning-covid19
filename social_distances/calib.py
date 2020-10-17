import numpy as np
import cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_perspective_transform(image, pts):
    # rect = order_points(pts)
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped_image, M


# def get_perspective_transform(image, pts):
# 	IMAGE_H = 720
# 	IMAGE_W = 1280
# 	src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# 	dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# 	# image = image[450:(450+IMAGE_H), 0:IMAGE_W]
# 	M = cv2.getPerspectiveTransform(src, dst)
# 	warped_image = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
# 	return warped_image, M


def real_world_scale(scale_points, perspective_data):
    dmx, dmy, M = (
        perspective_data["dmx"],
        perspective_data["dmx"],
        perspective_data["matrix"],
    )
    pt1 = np.array([[scale_points[0]]], dtype=np.float32)
    pt2 = np.array([[scale_points[1]]], dtype=np.float32)
    pt3 = np.array([[scale_points[2]]], dtype=np.float32)
    pt1_transformed = cv2.perspectiveTransform(pt1, M)[0][0]
    pt2_transformed = cv2.perspectiveTransform(pt2, M)[0][0]
    pt3_transformed = cv2.perspectiveTransform(pt3, M)[0][0]
    dh = np.sqrt(
        abs(pt1_transformed[0] - pt2_transformed[0]) ** 2
        + abs(pt1_transformed[1] - pt2_transformed[1]) ** 2
    )  # horizontal
    dv = np.sqrt(
        abs(pt1_transformed[0] - pt3_transformed[0]) ** 2
        + abs(pt1_transformed[1] - pt3_transformed[1]) ** 2
    )  # vertical
    scale_horizontal = float(dmx / dh)
    scale_vertical = float(dmy / dv)
    return scale_horizontal, scale_vertical


def get_transformed_points(boxes, perspective_data):
    """
    Hàm tính toán tâm đáy (centroid-bottom) cho tất cả các bbox và biến đổi phối cảnh bird-eye cho tất cả các điểm đó.
    """
    M = perspective_data["matrix"]
    foot_points = []
    for x1, y1, x2, y2, _ in boxes:
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cx_b, cy_b = cx, cy + (h // 2)
        pnts = np.array([[[cx_b, cy_b]]], dtype=np.float32)
        pnts_transformed = cv2.perspectiveTransform(pnts, M)[0][0]
        pnt = (int(pnts_transformed[0]), int(pnts_transformed[1]))
        foot_points.append(pnt)

    return foot_points
