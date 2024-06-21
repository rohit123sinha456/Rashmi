import numpy as np
import supervision as sv
from ultralytics import YOLO
from inference.models.yolo_world.yolo_world import YOLOWorld

# model = YOLOWorld(model_id="yolo_world/l")
# classes = ["blue and white bag"]
# model.set_classes(classes)

model = YOLO("./weightsv1/helmet.pt")
tracker = sv.ByteTrack() # track_activation_threshold=0.002,minimum_matching_threshold=0.001,frame_rate=25,minimum_consecutive_frames=1
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoundingBoxAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    print(frame.shape)
    results = model.predict(frame,conf=0.50)[0] # model.infer(frame,confidence=0.002)# 
    # detections = sv.Detections.from_inference(results).with_nms(threshold=0.8)
    detections = sv.Detections.from_ultralytics(results)
    trackdetections = tracker.update_with_detections(detections)
    annotated_image = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=trackdetections)
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(trackdetections.class_id, trackdetections.tracker_id)
    ]
    print(detections)
    print(trackdetections)
    print(labels)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=trackdetections, labels=labels)
    return annotated_image

sv.process_video(
    source_path="helmet.mp4",
    target_path="helmet_detect.mp4",
    callback=callback
)