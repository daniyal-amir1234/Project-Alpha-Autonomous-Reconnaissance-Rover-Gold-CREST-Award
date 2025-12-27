import cv2
import numpy as np

from tflite_support.task import core, processor, vision

from src.config import AppConfig
from src.utils.visualization import visualize_detections


class ObjectDetector:
    def __init__(self, detector: vision.ObjectDetector):
        self.detector = detector

    def detect_and_annotate(self, bgr_image: np.ndarray):
        # Convert BGR -> RGB as required by TF Lite Task vision APIs
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = self.detector.detect(input_tensor)

        annotated = visualize_detections(bgr_image, detection_result)
        return annotated, detection_result


def create_detector(cfg: AppConfig) -> ObjectDetector:
    base_options = core.BaseOptions(
        file_name=cfg.model_path,
        num_threads=cfg.num_threads
    )

    detection_options = processor.DetectionOptions(
        max_results=cfg.max_results,
        score_threshold=cfg.score_threshold
    )

    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        detection_options=detection_options
    )

    detector = vision.ObjectDetector.create_from_options(options)
    return ObjectDetector(detector)
