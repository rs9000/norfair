import argparse
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision.ops.boxes as bops
import yolov5

import norfair
from norfair import Detection, Tracker, Video

# We want to use IoU of 0.3 which is 1/0.3=3.33
DISTANCE_THRESHOLD_BBOX: float = 3.33
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10_000


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = yolov5.load(model_path, device=device)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def iou_pytorch(detection, tracked_object):
    # Slower but simplier version of iou

    detection_points = np.concatenate([detection.points[0], detection.points[1]])
    tracked_object_points = np.concatenate(
        [tracked_object.estimate[0], tracked_object.estimate[1]]
    )

    box1 = torch.tensor([detection_points], dtype=torch.float)
    box2 = torch.tensor([tracked_object_points], dtype=torch.float)
    iou = bops.box_iou(box1, box2)

    # Since 0 <= IoU <= 1
    # We define 1/IoU as a distance since this works as a distance.
    # If IoU ~ 0 then distance will be big and if IoU is near 1, distance will be 1
    # which is the minimum distance.
    return np.float(1 / iou if iou else MAX_DISTANCE)


def iou(detection, tracked_object):
    # Detection points will be box A
    # Tracked objects point will be box B.

    boxA = np.concatenate([detection.points[0], detection.points[1]])
    boxB = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Since 0 <= IoU <= 1
    # We define 1/IoU as a distance since this works as a distance.
    # If IoU ~ 0 then distance will be big and if IoU is near 1, distance will be 1
    # which is the minimum distance.
    return 1 / iou if iou else (MAX_DISTANCE)


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(Detection(points=centroid, scores=scores))
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(Detection(points=bbox, scores=scores))

    return norfair_detections


parser = argparse.ArgumentParser(description="Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument(
    "--detector_path", type=str, default="yolov5m6.pt", help="YOLOv5 model path"
)
parser.add_argument(
    "--img_size", type=int, default="720", help="YOLOv5 inference size (pixels)"
)
parser.add_argument(
    "--conf_thres",
    type=float,
    default="0.25",
    help="YOLOv5 object confidence threshold",
)
parser.add_argument(
    "--iou_thresh", type=float, default="0.45", help="YOLOv5 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track_points",
    type=str,
    default="centroid",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)

for input_path in args.files:
    video = Video(input_path=input_path)

    distance_function = iou if args.track_points == "bbox" else euclidean_distance
    distance_threshold = (
        DISTANCE_THRESHOLD_BBOX
        if args.track_points == "bbox"
        else DISTANCE_THRESHOLD_CENTROID
    )

    tracker = Tracker(
        distance_function=distance_function,
        distance_threshold=distance_threshold,
    )

    for frame in video:
        yolo_detections = model(
            frame,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thresh,
            image_size=args.img_size,
            classes=args.classes,
        )
        detections = yolo_detections_to_norfair_detections(
            yolo_detections, track_points=args.track_points
        )
        tracked_objects = tracker.update(detections=detections)
        if args.track_points == "centroid":
            norfair.draw_tracked_objects(frame, tracked_objects)
        elif args.track_points == "bbox":
            norfair.draw_tracked_boxes(frame, tracked_objects, border_width=2)
        video.write(frame)
