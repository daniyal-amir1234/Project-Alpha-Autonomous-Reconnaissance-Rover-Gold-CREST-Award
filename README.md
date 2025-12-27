# Project Alpha — Autonomous Reconnaissance Rover
Sep 2023 – Jul 2024

## Overview
**Awarded: Gold CREST Award (Engineering Education Scheme)**

Project Alpha is an autonomous reconnaissance rover developed as part of the Engineering Education Scheme (EES) and awarded the Gold CREST Award. The system combines a manually driven rover platform with an AI-powered sentry turret, capable of real-time object detection and tracking using TensorFlow Lite and OpenCV running on a Raspberry Pi. The turret uses stepper motors for pan/tilt actuation and a USB webcam for live vision input.

In a multidisciplinary team of 7 engineers, computer scientists and physicists, we built a reconnaissance rover with a camera 'sentry' turret: the rover platform was manually controlled, while the turret performed real-time object detection and basic target tracking using TensorFlow Lite + OpenCV on a Raspberry Pi.

## Features
- Real-time object detection (TensorFlow Lite Task Library)
- Webcam / Pi camera capture (OpenCV)
- FPS counter overlay (matches recovered screenshot structure)
- Optional pan/tilt tracking interface (GPIO stepper control stubbed + mock driver)

## Project Goals
- Reduce human risk in dangerous environments by enabling remote reconnaissance.
- Perform on-device AI inference on low-power hardware.
- Track detected objects using a motorised pan/tilt turret.
- Design and deliver a complex system using formal engineering planning (Gantt charts, milestones, modular design).
- Formally document this process in a professionally written technical portfolio and report.

## System Architecture
USB Webcam
    ↓
OpenCV Video Capture
    ↓
TensorFlow Lite Object Detector
    ↓
Bounding Box + Confidence Scores
    ↓
Target Selection Logic
    ↓
Pan/Tilt Controller (Stepper Motors)
    ↓
Fire sentry turret (not actually created)

- Raspberry Pi acts as the central compute unit
- USB webcam provides live RGB video
- TensorFlow Lite Task Library performs object detection
- OpenCV handles image capture, colour conversion, and visualisation
- Stepper motors drive the sentry turret via GPIO
- FPS monitoring ensures real-time performance visibility


## Tech
### Software
- Python
- OpenCV
- TensorFlow Lite (tflite-support Task API)
- NumPy

### Hardware
- Raspberry Pi
- USB Webcam
- Stepper motors (pan + tilt)
- Stepper motor drivers (e.g. A4988 / DRV8825)
- Rover chassis (rocker-bogie inspired)

## Repository Structure
```
project-alpha-sentry-rover/
├─ src/
│  ├─ main.py            # Main capture + inference loop
│  ├─ detector.py        # TensorFlow Lite object detector
│  ├─ camera.py          # USB webcam interface
│  ├─ tracking.py        # Target selection & offset calculation
│  ├─ config.py          # Centralised configuration
│  └─ utils/
│     ├─ fps.py          # FPS averaging logic
│     └─ visualization.py
├─ hardware/
│  ├─ pantilt.py         # Pan/tilt controller abstraction
│  ├─ stepper_gpio.py    # GPIO stepper driver
│  └─ mock.py            # Non-GPIO mock controller
├─ assets/
│  ├─ models/            # TensorFlow Lite models
│  └─ screenshots/       # Recovered original code screenshots
├─ docs/
│  ├─ Portfolio_Report.pdf
│  └─ Design_Brief.docx
├─ requirements.txt
├─ requirements-pi.txt
└─ README.md
```

## Quick start (Laptop / Desktop)
```
python -m venv .venv
source .venv/bin/activate  # if you have Windows, do: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.main --model assets/models/efficientdet_lite0.tflite
```
