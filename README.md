#  Real-Time Face Recognition & Attendance System

## Overview

A production-style real-time face recognition system designed for low-latency inference using RTSP streams and GPU acceleration.

##  Architecture

RTSP Stream → Capture Thread → Inference Thread → API → UI

* Latest-frame processing (no backlog)
* Asynchronous pipeline
* Modular services

##  Features

* Real-time face detection & recognition
* GPU acceleration (ONNX Runtime / InsightFace)
* Attendance logging (SQLite + snapshots)
* Anti-spoofing integration (optional)
* Multi-threaded architecture
* Low-latency processing (<100ms backend)

##  Key Challenges Solved

* Eliminated frame buffering lag
* Fixed embedding mismatch issues
* Optimized inference pipeline
* Stabilized RTSP streaming
* Debugged environment + dependency issues

##  Project Structure

```
config.py
state.py
capture_service.py
inference_service.py
attendance.py
app.py
```

##  How to Run

```bash
git clone <repo>
cd face-recognition
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

##  Notes

* Dataset quality directly impacts recognition accuracy
* MJPEG used for preview (WebRTC planned for production UI)

##  Future Improvements

* WebRTC streaming (ultra-low latency UI)
* Face tracking optimization
* Multi-camera support
* REST API for attendance dashboards

