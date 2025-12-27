import argparse
import sys

import cv2

from src.config import AppConfig
from src.camera import open_camera
from src.detector import create_detector
from src.utils.fps import FpsCounter
from src.utils.visualization import draw_fps

from src.tracking import TrackingController


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(description="Project Alpha - Object Detection + Optional Pan/Tilt Tracking")

    p.add_argument("--model", required=True, help="Path to a .tflite model (e.g., EfficientDet Lite).")
    p.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    p.add_argument("--max-results", type=int, default=3)
    p.add_argument("--score-threshold", type=float, default=0.30)
    p.add_argument("--num-threads", type=int, default=2)

    p.add_argument("--use-gpio", type=int, default=0, help="1 to enable GPIO stepper output (Pi only).")
    p.add_argument("--mirror", type=int, default=1, help="1 to flip horizontally like a selfie cam.")

    return AppConfig(
        model_path=p.parse_args().model,
        camera_id=p.parse_args().camera_id,
        width=p.parse_args().width,
        height=p.parse_args().height,
        max_results=p.parse_args().max_results,
        score_threshold=p.parse_args().score_threshold,
        num_threads=p.parse_args().num_threads,
        use_gpio=bool(p.parse_args().use_gpio),
        mirror=bool(p.parse_args().mirror),
    )


def main():
    cfg = parse_args()

    cap = open_camera(cfg.camera_id, cfg.width, cfg.height)
    detector = create_detector(cfg)

    tracker = TrackingController(use_gpio=cfg.use_gpio)
    fps_counter = FpsCounter(avg_over_n_frames=10)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ERROR: Unable to read from webcam. Please verify your webcam settings.", file=sys.stderr)
            sys.exit(1)

        if cfg.mirror:
            image = cv2.flip(image, 1)

        # Run detection (detector handles RGB conversion internally)
        annotated, detection_result = detector.detect_and_annotate(image)

        # Optional: use detections to compute pan/tilt adjustments
        tracker.update(detection_result, frame_shape=image.shape)

        # FPS overlay (matches screenshot approach: recompute every N frames)
        fps = fps_counter.tick()
        annotated = draw_fps(annotated, fps)

        cv2.imshow("project_alpha_detector", annotated)

        # ESC to quit
        if cv2.waitKey(1) == 27:
            break

    tracker.shutdown()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
