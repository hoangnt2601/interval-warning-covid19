import cv2
import numpy as np
from camera.video import FileVideoStream, VideoStream
from social_distances import dist, plot, calib
import onnxruntime as ort
import torch
from utils.torch_utils import time_synchronized
from utils.general import non_max_suppression
from utils.process import preprocess
from camera.fps import FPS
from tracker import sort
import json


if __name__ == "__main__":
    fps = FPS().start()
    vs = FileVideoStream("../video/pedestrians.mp4").start()
    tracker = sort.SortTracker()

    resize_w, resize_h = dist.get_scale(vs.width, vs.height)
    # out = cv2.VideoWriter('meditech.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (vs.width, vs.height))

    # config model
    img_size = 640
    input_names = ["images"]
    output_names = ["outputs"]
    conf_thres = 0.35
    iou_thres = 0.5
    classes = 0
    # load model
    session = ort.InferenceSession("yolov5s.onnx")

    perspective_data = {}
    with open("coords.json", "r") as f:
        coords = json.load(f)
        offsets = np.array(coords["offsets"], dtype=np.float32)
        scale_points = np.array(coords["scales"], dtype=np.float32)

    while True:
        frame = vs.read()
        if frame is None:
            break

        H, W = frame.shape[:2]

        # detect person
        input_, scale = preprocess(frame, img_size)
        input_ = np.expand_dims(input_, 0)

        boxes = []

        t0 = time_synchronized()
        try:
            outputs = session.run(output_names, {input_names[0]: input_})
            dets = non_max_suppression(
                torch.tensor(outputs[0]),
                conf_thres,
                iou_thres,
                classes=classes,
                agnostic=True,
            )[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))
        t1 = time_synchronized()

        if dets is None:
            continue
        if dets.nelement() == 0:
            continue
        for det in dets:
            det = det.cpu().detach().numpy()
            x1, y1, x2, y2, score, label = (det / scale).astype(np.int32)
            if label != 0:
                continue
            boxes.append([x1, y1, x2, y2])

        tracked_boxes = tracker.update(np.array(boxes))

        centroids = []
        for tracked_box in tracked_boxes:
            x1, y1, x2, y2, id_obj = tracked_box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0),2)

        _, perspective_matrix = calib.get_perspective_transform(frame, offsets)
        perspective_data["dmx"] = 1
        perspective_data["dmy"] = 2
        perspective_data["matrix"] = perspective_matrix
        # Lấy điểm trung tâm dưới chân của obj theo tỉ lệ transform
        birdeye_points = calib.get_transformed_points(tracked_boxes, perspective_data)
        scale_w, scale_h = calib.real_world_scale(scale_points, perspective_data)
        perspective_data["scale_wh"] = [scale_w, scale_h]
        # Tính khoảng cách và trạng thái của các obj
        # print(len(centroids), len(birdeye_points))
        birdeye_centroids, camera_centroids = dist.get_distances(
            centroids, birdeye_points, perspective_data, threshold=2
        )
        real_image = plot.social_distancing_view(frame, camera_centroids)

        fps.update()
        fps_text = fps.get_fps_n()
        cv2.putText(
            frame,
            "FPS: {}".format(round(fps_text, 0)),
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Infer Time: {}ms".format(round((t1 - t0) * 1000, 0)),
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        # out.write(image)
        cv2.imshow("f", real_image)
        if cv2.waitKey(20) == ord("q"):
            break

    # out.release()
    vs.stop()
    cv2.destroyAllWindows()
