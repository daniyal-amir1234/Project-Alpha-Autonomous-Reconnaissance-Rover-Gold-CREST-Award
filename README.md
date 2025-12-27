# Project Alpha — Autonomous Reconnaissance Rover (Gold CREST Award)

A Raspberry Pi + Computer Vision project built for the Engineering Education Scheme (EES) from Sep 2023 – Jul 2024.

Awarded: Gold CREST Award (British Science Association).

We built a reconnaissance rover with a camera 'sentry' turret: the rover platform was manually controlled, while the turret performed real-time object detection and basic target tracking using TensorFlow Lite + OpenCV on a Raspberry Pi.


## Features
- Real-time object detection (TensorFlow Lite Task Library)
- Webcam / Pi camera capture (OpenCV)
- FPS counter overlay (matches recovered screenshot structure)
- Optional pan/tilt tracking interface (GPIO stepper control stubbed + mock driver)

## Tech
- Python
- OpenCV
- TensorFlow Lite / tflite-support (Task library)
- Raspberry Pi GPIO

## Quick start (Laptop / Desktop)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.main --model assets/models/efficientdet_lite0.tflite
