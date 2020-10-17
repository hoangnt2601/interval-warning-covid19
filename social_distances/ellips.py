import numpy as np
import cv2
from itertools import product


def perspective_transform(src, M):

    # first we extend source points by a column of 1
    # augment has shape (n,1)
    augment = np.ones((src.shape[0], 1))
    # projective_corners is a 3xn matrix with last row all 1
    # note that we transpose the concatenation
    projective_corners = np.concatenate((src, augment), axis=1).T

    # projective_points has shape 3xn
    projective_points = M.dot(projective_corners)

    # obtain the target_points by dividing the projective_points
    # by its last row (where it is non-zero)
    # target_points has shape (3,n).
    target_points = np.true_divide(projective_points, projective_points[-1])

    # so we want return points in row form
    return target_points[:2].T


def evaluate_ellipses(boxes, ellipses_boxes_draw, ellipses_boxes, scaling_factor, M):
    for box in boxes:
        xmin, ymin, xmax, ymax, _ = box
        cx, cy = int((xmax + xmin) / 2), int((ymax + ymin) / 2)
        hbox = ymax - ymin
        ellip_height = int(round(hbox * scaling_factor, 2))

        # get translation point
        src = np.array([[cx, ymin], [cx, ymax]], np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, M)
        ellip_width = int(dst[1, 0][1] - dst[0, 0][1])

        # Bounding box surrounding the ellipses, useful to compute whether there is any overlap between two ellipses
        # ellipses_boxes.append([cx-ellip_width, cx+ellip_width, cy-ellip_height, cy+ellip_height])
        ellipses_boxes.append(
            [cx - ellip_width, cy - ellip_height, cx + ellip_width, cy + ellip_height]
        )
        # for draw
        ellipses_boxes_draw.append([cx, cy, ellip_width, ellip_height])


# def iou(rect1, rect2):
# 	if (rect1[0] >= rect2[1] or rect2[0] >= rect1[1]):
# 		return False

# 	if (rect1[3] <= rect2[2] or rect2[3] <= rect1[2]):
# 		return False

# 	return True


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def evaluate_overlapping(ellipses_boxes, tracked_boxes, warning_boxes):
    list_indexes = np.arange(len(ellipses_boxes)).tolist()
    for i, j in product(list_indexes, list_indexes):
        perA = ellipses_boxes[i]
        perB = ellipses_boxes[j]
        if i != j:
            if iou(perA, perB) > 0:
                warning_boxes.append([tracked_boxes[i], tracked_boxes[j], True])
            else:
                warning_boxes.append([tracked_boxes[i], tracked_boxes[j], False])


def trace(frame, warning_boxes, pad=3, sizec=2):
    for box in warning_boxes:
        # cxA, cyA = int((box[0][0] + box[0][2])/2), int((box[0][1] + box[0][3])/2)
        # cxB, cyB = int((box[1][0] + box[1][2])/2), int((box[1][1] + box[1][3])/2)
        x1A, y1A, x2A, y2A, idA = box[0][0], box[0][1], box[0][2], box[0][3], box[0][4]
        cxA, cyA = int((x1A + x2A) / 2), int((y1A + y2A) / 2)
        x1B, y1B, x2B, y2B, idB = box[1][0], box[1][1], box[1][2], box[1][3], box[1][4]
        cxB, cyB = int((x1B + x2B) / 2), int((y1B + y2B) / 2)
        warning = box[2]

        if warning:
            cv2.rectangle(frame, (x1A, y1A), (x2A, y2A), (0, 0, 255), sizec)
            cv2.rectangle(frame, (x1B, y1B), (x2B, y2B), (0, 0, 255), sizec)
            cv2.putText(
                frame,
                "Waring ID {}".format(idA),
                (x1A, y1A - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                sizec,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Waring ID {}".format(idB),
                (x1B, y1B - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                sizec,
                cv2.LINE_AA,
            )
            cv2.circle(frame, (cxA, cyA), pad, (0, 255, 255), pad)
            cv2.circle(frame, (cxB, cyB), pad, (0, 255, 255), pad)
            cv2.line(
                frame,
                (cxA - pad, cyA - pad),
                (cxB - pad, cyB - pad),
                (0, 0, 255),
                sizec,
            )


def draw(frame, ellipses_boxes_draw):
    for box in ellipses_boxes_draw:
        # Trace ellipse
        cv2.ellipse(
            frame, (box[0], box[1]), (box[2], box[3]), 0, 0, 360, (255, 0, 255), 2
        )
