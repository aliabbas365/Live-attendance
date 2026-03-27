# Fixed main.py (cleaned and working version)

import cv2
import time
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

BASE_DIR = Path(".")
CAMERA_SOURCE = "/FACE-RECOGNITION/test_video.mp4"

# -------------------- GPU MODEL --------------------
print("Loading InsightFace model...")
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(512, 512))
print("Model ready")

# -------------------- TRACKER FIX --------------------
def create_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise Exception("No tracker available")

# -------------------- SIMPLE TRACK --------------------
class Track:
    def __init__(self, frame, bbox):
        self.tracker = create_tracker()
        self.tracker.init(frame, tuple(map(int, bbox)))
        self.bbox = bbox

    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        if ok:
            self.bbox = bbox
        return ok, self.bbox

# -------------------- PIPELINE --------------------
class SimplePipeline:
    def __init__(self):
        self.frame_index = 0
        self.tracks = []

    def process(self, frame):
        self.frame_index += 1
        faces = app.get(frame)

        annotated = frame.copy()

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

        return annotated

# -------------------- MAIN --------------------
def start_recognition():
    pipeline = SimplePipeline()

    source_str = str(CAMERA_SOURCE).lower()
    is_video = source_str.endswith((".mp4", ".avi", ".mov"))

    if is_video:
        print("Starting video:", CAMERA_SOURCE)
        cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not cap.isOpened():
            print("Cannot open video")
            return

        width = int(cap.get(3))
        height = int(cap.get(4))

        out = cv2.VideoWriter(
            "output_test.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (width, height)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            annotated = pipeline.process(frame)
            out.write(annotated)

            print("Frame:", pipeline.frame_index)

        cap.release()
        out.release()

    else:
        print("Starting live stream:", CAMERA_SOURCE)
        cap = cv2.VideoCapture(CAMERA_SOURCE)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            annotated = pipeline.process(frame)
            print("Frame:", pipeline.frame_index)

if __name__ == "__main__":
    start_recognition()
