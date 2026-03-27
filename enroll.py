import os
import cv2
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from insightface.app import FaceAnalysis

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_FILE = BASE_DIR / "enrolled_embeddings.pkl"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
UNKNOWN_DIR = BASE_DIR / "unknown_faces"
LOG_DIR = BASE_DIR / "attendance_logs"

SNAPSHOT_DIR.mkdir(exist_ok=True)
UNKNOWN_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

RECOGNITION_THRESHOLD = 0.45
CAMERA_INDEX = 0
TARGET_WIDTH = 960
DETECT_EVERY_N_FRAMES = 3
RECOGNIZE_EVERY_N_FRAMES = 5
TRACK_TTL = 10
MAX_NUM_FACES = 5
SHOW_ROI = False

# ROI as percentages
ROI_X1 = 15
ROI_Y1 = 5
ROI_X2 = 85
ROI_Y2 = 95

# ============================================================
# ROI MANAGER
# ============================================================

class ROIManager:
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_polygon = None
        self.roi_enabled = False

    def set_frame_size(self, width, height):
        self.frame_width = width
        self.frame_height = height

    def set_roi_rectangle(self, x1_pct, y1_pct, x2_pct, y2_pct):
        x1 = int(x1_pct / 100 * self.frame_width)
        y1 = int(y1_pct / 100 * self.frame_height)
        x2 = int(x2_pct / 100 * self.frame_width)
        y2 = int(y2_pct / 100 * self.frame_height)

        self.roi_polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32
        )
        self.roi_enabled = True

    def disable_roi(self):
        self.roi_enabled = False

    def is_in_roi(self, bbox):
        if not self.roi_enabled or self.roi_polygon is None:
            return True

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        return cv2.pointPolygonTest(
            self.roi_polygon,
            (float(center_x), float(center_y)),
            False
        ) >= 0

    def draw_roi(self, frame, color=(0, 255, 0), thickness=2):
        if self.roi_enabled and self.roi_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [self.roi_polygon], True, color, thickness)
            cv2.putText(
                frame,
                "ENTRANCE ROI",
                tuple(self.roi_polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        return frame

# ============================================================
# ATTENDANCE LOGGER
# ============================================================

class AttendanceLogger:
    def __init__(self, cooldown_seconds=300):
        self.cooldown_seconds = cooldown_seconds
        self.last_seen = {}
        self.daily_records = {}
        self.all_logs = []
        self.unknown_count = 0
        self.today = datetime.now().strftime("%Y-%m-%d")

    def _is_on_cooldown(self, emp_id):
        if emp_id not in self.last_seen:
            return False
        return (time.time() - self.last_seen[emp_id]) < self.cooldown_seconds

    def _save_crop(self, frame, bbox, save_path):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        pad = 20
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(str(save_path), crop)

    def log_recognition(self, emp_id, name, confidence, frame, bbox):
        if self._is_on_cooldown(emp_id):
            return False, "Cooldown"

        now_ts = time.time()
        now_iso = datetime.fromtimestamp(now_ts).isoformat()
        self.last_seen[emp_id] = now_ts

        event_type = "CHECK_IN"
        if emp_id in self.daily_records:
            self.daily_records[emp_id]["check_out"] = now_ts
            self.daily_records[emp_id]["detections"] += 1
            event_type = "CHECK_OUT"
        else:
            self.daily_records[emp_id] = {
                "name": name,
                "check_in": now_ts,
                "check_out": None,
                "detections": 1,
            }

        filename = f"{emp_id}_{datetime.fromtimestamp(now_ts).strftime('%Y%m%d_%H%M%S')}.jpg"
        self._save_crop(frame, bbox, SNAPSHOT_DIR / filename)

        self.all_logs.append({
            "emp_id": emp_id,
            "name": name,
            "event_type": event_type,
            "confidence": round(float(confidence), 4),
            "timestamp": now_iso,
            "snapshot": filename
        })
        return True, event_type

    def log_unknown(self, frame, bbox, score):
        self.unknown_count += 1
        now_ts = time.time()
        filename = f"unknown_{self.unknown_count:04d}_{datetime.fromtimestamp(now_ts).strftime('%Y%m%d_%H%M%S')}.jpg"
        self._save_crop(frame, bbox, UNKNOWN_DIR / filename)

    def export_csv(self):
        import pandas as pd
        df = pd.DataFrame(self.all_logs)
        path = LOG_DIR / f"attendance_{self.today}.csv"
        df.to_csv(path, index=False)
        return path

# ============================================================
# RECOGNITION ENGINE
# ============================================================

class FaceRecognitionEngine:
    def __init__(self, enrolled_embeddings, employee_db, threshold=0.45):
        self.enrolled_embeddings = enrolled_embeddings
        self.employee_db = employee_db
        self.threshold = threshold
        self.ids = list(enrolled_embeddings.keys())
        self.matrix = np.stack([enrolled_embeddings[k] for k in self.ids]).astype(np.float32)

    def identify(self, embedding):
        scores = np.dot(self.matrix, embedding)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_id = self.ids[best_idx]

        if best_score >= self.threshold:
            name = self.employee_db[best_id]["name"]
            return best_id, name, best_score
        return None, "Unknown", best_score

# ============================================================
# FAST TRACKING PIPELINE
# ============================================================

class Track:
    def __init__(self, track_id, bbox, frame):
        self.track_id = track_id
        self.bbox = bbox
        self.name = "Unknown"
        self.emp_id = None
        self.confidence = 0.0
        self.embedding = None
        self.last_seen = time.time()
        self.last_recognized = 0
        self.missed_frames = 0
        self.history = deque(maxlen=10)

        self.tracker = cv2.TrackerCSRT_create()
        x, y, w, h = self.xyxy_to_xywh(bbox)
        self.tracker.init(frame, (x, y, w, h))

    @staticmethod
    def xyxy_to_xywh(bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def xywh_to_xyxy(box):
        x, y, w, h = [int(v) for v in box]
        return [x, y, x + w, y + h]

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if ok:
            self.bbox = self.xywh_to_xyxy(box)
            self.last_seen = time.time()
            self.history.append(self.bbox)
        return ok

class FastFacePipeline:
    def __init__(self, face_app, recognition_engine, roi_manager, logger):
        self.face_app = face_app
        self.recognition_engine = recognition_engine
        self.roi = roi_manager
        self.logger = logger

        self.frame_index = 0
        self.next_track_id = 1
        self.tracks = {}

    def iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = max(1, (a[2]-a[0]) * (a[3]-a[1]))
        areaB = max(1, (b[2]-b[0]) * (b[3]-b[1]))
        return inter / float(areaA + areaB - inter)

    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        self.roi.set_frame_size(w, h)
        if SHOW_ROI and self.roi.roi_polygon is None:
            self.roi.set_roi_rectangle(ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
        elif not SHOW_ROI:
            self.roi.disable_roi()

        if w <= TARGET_WIDTH:
            return frame, 1.0

        scale = TARGET_WIDTH / w
        resized = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))
        return resized, scale

    def scale_back(self, bbox, scale):
        if scale == 1.0:
            return bbox
        return [int(v / scale) for v in bbox]

    def detect(self, frame_small):
        faces = self.face_app.get(frame_small, max_num=MAX_NUM_FACES)
        detections = []
        for face in faces:
            detections.append({
                "bbox": [int(v) for v in face.bbox],
                "embedding": face.normed_embedding
            })
        return detections

    def match_tracks(self, detections, frame_full, scale):
        matched = set()
        unmatched_tracks = set(self.tracks.keys())

        for i, det in enumerate(detections):
            det_full = self.scale_back(det["bbox"], scale)
            best_tid = None
            best_iou = 0.0

            for tid, tr in self.tracks.items():
                score = self.iou(det_full, tr.bbox)
                if score > best_iou and score >= 0.3:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None:
                tr = self.tracks[best_tid]
                tr.bbox = det_full
                tr.embedding = det["embedding"]
                tr.missed_frames = 0
                tr.last_seen = time.time()

                tr.tracker = cv2.TrackerCSRT_create()
                x, y, w, h = Track.xyxy_to_xywh(det_full)
                tr.tracker.init(frame_full, (x, y, w, h))

                matched.add(i)
                unmatched_tracks.discard(best_tid)

        for i, det in enumerate(detections):
            if i in matched:
                continue
            det_full = self.scale_back(det["bbox"], scale)
            tr = Track(self.next_track_id, det_full, frame_full)
            tr.embedding = det["embedding"]
            self.tracks[self.next_track_id] = tr
            self.next_track_id += 1

        for tid in unmatched_tracks:
            self.tracks[tid].missed_frames += 1

        dead = [tid for tid, tr in self.tracks.items() if tr.missed_frames > TRACK_TTL]
        for tid in dead:
            del self.tracks[tid]

    def update_trackers(self, frame):
        dead = []
        for tid, tr in self.tracks.items():
            ok = tr.update(frame)
            if not ok:
                tr.missed_frames += 1
            if tr.missed_frames > TRACK_TTL:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def recognize_tracks(self, frame):
        now = time.time()
        for tid, tr in self.tracks.items():
            if tr.embedding is None:
                continue

            if now - tr.last_recognized < 1.0:
                continue

            if not self.roi.is_in_roi(tr.bbox):
                tr.emp_id = None
                tr.name = "Outside ROI"
                tr.confidence = 0.0
                continue

            emp_id, name, conf = self.recognition_engine.identify(tr.embedding)
            tr.emp_id = emp_id
            tr.name = name if emp_id is not None else "Unknown"
            tr.confidence = conf
            tr.last_recognized = now

            if emp_id is not None:
                self.logger.log_recognition(emp_id, name, conf, frame, tr.bbox)

    def draw(self, frame):
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]

            if tr.name == "Outside ROI":
                color = (150, 150, 150)
                label = "Outside ROI"
            elif tr.emp_id is None:
                color = (0, 0, 255)
                label = f"Unknown {tr.confidence:.2f}"
            else:
                color = (0, 255, 0)
                label = f"{tr.name} {tr.confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            top = max(0, y1 - text_size[1] - 10)
            cv2.rectangle(frame, (x1, top), (x1 + text_size[0] + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        frame = self.roi.draw_roi(frame)
        return frame

    def process(self, frame):
        self.frame_index += 1
        frame_small, scale = self.resize_frame(frame)

        if self.frame_index % DETECT_EVERY_N_FRAMES == 1:
            detections = self.detect(frame_small)
            self.match_tracks(detections, frame, scale)
        else:
            self.update_trackers(frame)

        if self.frame_index % RECOGNIZE_EVERY_N_FRAMES == 1:
            self.recognize_tracks(frame)

        return self.draw(frame.copy())

# ============================================================
# LOAD DATA
# ============================================================

if not EMBEDDINGS_FILE.exists():
    raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_FILE}")

with open(EMBEDDINGS_FILE, "rb") as f:
    enrolled_embeddings = pickle.load(f)

employee_db = {
    emp_id: {"name": emp_id}
    for emp_id in enrolled_embeddings.keys()
}

# ============================================================
# INIT MODEL
# ============================================================

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

roi_manager = ROIManager()
if SHOW_ROI:
    roi_manager.set_roi_rectangle(ROI_X1, ROI_Y1, ROI_X2, ROI_Y2)
else:
    roi_manager.disable_roi()

attendance_logger = AttendanceLogger(cooldown_seconds=300)
recognition_engine = FaceRecognitionEngine(enrolled_embeddings, employee_db, threshold=RECOGNITION_THRESHOLD)
pipeline = FastFacePipeline(app, recognition_engine, roi_manager, attendance_logger)

# ============================================================
# MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

prev = time.time()
print("Press Q to quit, S to save CSV log.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = pipeline.process(frame)

    now = time.time()
    fps = 1.0 / max(1e-6, now - prev)
    prev = now

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.imshow("Face Attendance - Real Time", annotated)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        path = attendance_logger.export_csv()
        print(f"Saved log to {path}")

cap.release()
cv2.destroyAllWindows()