"""
Contain functions to draw Bird Eye View for region of interest(ROI) and draw bounding boxes according to risk factor
for humans in a frame and draw lines between boxes according to risk factor between two humans. 
"""

import cv2
import numpy as np


def bird_eye_view(
    frame, birdeye_centroids, birdeye_points, resize_w, resize_h, sizec=5
):
    """
    Chức năng vẽ Bird-Eye cho vùng offset
    Params:
        targets: Cặp các obj thoả/không thoả điều kiện giãn cách 2m trên tỉ lệ birdeye
        resize_w, resize_h: Tỉ lệ khung hinh birdeye với frame
    """
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)
    background = (200, 200, 200)

    blank_image = np.zeros((int(h * resize_h), int(w * resize_w), 3), np.uint8)
    blank_image[:] = background

    for target in birdeye_centroids:
        personA = tuple(
            map(int, np.array(target[0]) * [resize_w, resize_h])
        )  # centroid
        personB = tuple(
            map(int, np.array(target[1]) * [resize_w, resize_h])
        )  # centroid
        warning = target[2]

        if warning:
            cv2.circle(blank_image, personA, sizec, red, sizec)
            cv2.circle(blank_image, personB, sizec, red, sizec)
            blank_image = cv2.line(blank_image, personA, personB, red, sizec)
        else:
            cv2.circle(blank_image, personA, sizec, green, sizec)
            cv2.circle(blank_image, personB, sizec, green, sizec)

    return blank_image


def social_distancing_view(frame, targets, sizec=10):
    """
    Hiển thị mức độ vi phạm giữa các đối tượng, đỏ là vi phạm ngược lại là xanh
    Params:
        targets: Cặp các obj thoả/không thoả điều kiện giãn cách 2m
    """
    red = (0, 0, 255)
    green = (0, 255, 0)

    for target in targets:
        personA = target[0]  # centroid
        personB = target[1]  # centroid
        warning = target[2]

        if warning:
            cv2.circle(frame, personA, sizec, red, sizec)
            cv2.circle(frame, personB, sizec, red, sizec)
            frame = cv2.line(frame, personA, personB, red, sizec)
        else:
            cv2.circle(frame, personA, sizec, green, sizec)
            cv2.circle(frame, personB, sizec, green, sizec)

    return frame


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Scale and image keeping the proportion of it, for example if width is
    None but height setted tiy get an image of height size but keeping the
    width proportion. If width and height both are setted you get an image
    keeping the original aspect ration centered in the new image.
    Args:
        img: 'cv2.math' image to resize
        width: 'int' new width size to resize
        height: 'int' new height size to resize
    returns:
    """

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # resize the image
    if width is not None and height is not None:
        background = np.zeros((height, width, 3), np.uint8)
        y_pos = int((height * 0.5) - (resized.shape[0] * 0.5))
        background[y_pos : y_pos + resized.shape[0], 0 : resized.shape[1]] = resized
        return background

    # return the resized image
    return resized


def top_view(warped_image, max_w=500, max_h=500):
    warped_img_width = warped_image.shape[1]
    warped_img_height = warped_image.shape[0]

    hfscl = warped_img_width / max_w
    vfscl = warped_img_height / max_h

    if vfscl > hfscl:
        top_view_img = image_resize(
            image=warped_image,
            width=None,
            height=int(max_w),
            inter=cv2.INTER_AREA,
        )
    else:
        top_view_img = image_resize(
            image=warped_image,
            width=int(max_h),
            height=None,
            inter=cv2.INTER_AREA,
        )

    return top_view_img
