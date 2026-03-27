import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import sys
import time
import json
import pickle
import sqlite3
import warnings
import threading
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from collections import deque
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "employees_structured"
EMBEDDINGS_PATH = BASE_DIR / "embeddings" / "face_embeddings.pkl"

SNAPSHOTS_DIR = BASE_DIR / "snapshots"
UNKNOWN_DIR = BASE_DIR / "unknown_faces"
ATTENDANCE_DIR = BASE_DIR / "attendance_logs"
ANTI_SPOOF_MODEL_DIR = BASE_DIR / "resources" / "anti_spoof_models"
ANTI_SPOOF_DEVICE_ID = -1

os.makedirs(BASE_DIR / "embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

RECOGNITION_THRESHOLD = 0.38
MIN_FACE_SIZE = 15
MIN_DET_SCORE = 0.40
MIN_SHARPNESS = 2.0
MIN_INTEROCULAR_RATIO = 0.35

LOG_COOLDOWN_SECONDS = 300

TARGET_WIDTH = 1280
DETECT_EVERY_N_FRAMES = 5
RECOGNIZE_EVERY_N_FRAMES = 8
MAX_NUM_FACES = 3
TRACK_TTL = 10

CAMERA_SOURCE = "rtsp://Ali:Veeivs%232026@182.190.154.57:56471/Streaming/Channels/1001"

SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = BASE_DIR / "output_phase2.mp4"

# ============================================================
# SOURCE READER
# ============================================================

class VideoStream:
    def __init__(self, src):
        self.src = src
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self._open_stream()

    def _open_stream(self):
        if self.stream is not None:
            try:
                self.stream.release()
            except Exception:
                pass

        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()

        if self.grabbed:
            print("✅ RTSP stream connected")
        else:
            print("⚠️ RTSP stream connection failed, retrying...")

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if self.stream is None or not self.stream.isOpened():
                print("⚠️ Stream closed. Reconnecting...")
                time.sleep(2)
                self._open_stream()
                continue

            grabbed, frame = self.stream.read()

            if not grabbed or frame is None:
                print("⚠️ Frame read failed. Reconnecting RTSP...")
                time.sleep(2)
                self._open_stream()
                continue

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            if self.stream is not None:
                self.stream.release()
        except Exception:
            pass

# ============================================================
# ANTI-SPOOF
# ============================================================

class AntiSpoofEngine:
    def __init__(self, model_dir, device_id=-1):
        self.model_dir = str(model_dir)
        self.device_id = device_id
        self.enabled = False
        self.model_test = None
        self.image_cropper = None
        self._load_backend(device_id)

    def _load_backend(self, device_id):
        self.enabled = False
        self.device_id = device_id
        self.model_test = None
        self.image_cropper = None

        try:
            from src.anti_spoof_predict import AntiSpoofPredict
            from src.generate_patches import CropImage

            self.model_test = AntiSpoofPredict(device_id=device_id)
            self.image_cropper = CropImage()
            self.enabled = True

            if device_id >= 0:
                print(f"🛡️ Anti-spoof loaded (cuda:{device_id})")
            else:
                print("🛡️ Anti-spoof loaded (CPU)")
            return True

        except ImportError:
            print("⚠️ Anti-spoof src package not found. Liveness check disabled.")
            return False

        except Exception as e:
            if device_id >= 0:
                print(f"⚠️ Anti-spoof CUDA init failed: {e}. Falling back to CPU.")
                return self._load_backend(-1)

            print(f"⚠️ Anti-spoof failed to load on CPU: {e}. Liveness check disabled.")
            return False

    def _predict_once(self, frame, bbox):
        from src.anti_spoof_predict import parse_model_name

        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        image_bbox = [x1, y1, w, h]

        prediction = np.zeros((1, 3), dtype=np.float32)

        for model_name in sorted(os.listdir(self.model_dir)):
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.isfile(model_path):
                continue

            h_input, w_input, model_type, scale = parse_model_name(model_name)

            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False

            img = self.image_cropper.crop(**param)
            prediction += self.model_test.predict(img, model_path)

        label = int(np.argmax(prediction))
        return label == 1

    def is_real(self, frame, bbox):
        if not self.enabled:
            return None

        try:
            return self._predict_once(frame, bbox)

        except Exception as e:
            msg = str(e)

            if self.device_id >= 0 and "cuda" in msg.lower():
                print(f"⚠️ Anti-spoof CUDA runtime failed: {e}. Falling back to CPU.")
                if self._load_backend(-1):
                    try:
                        return self._predict_once(frame, bbox)
                    except Exception as e2:
                        print(f"Liveness check error after CPU fallback: {e2}")
                        return None

            print(f"Liveness check error: {e}")
            return None

# ============================================================
# ATTENDANCE LOGGER
# ============================================================

class AttendanceLogger:
    def __init__(self, cooldown_seconds=300):
        self.cooldown_seconds = cooldown_seconds
        self.last_seen = {}
        self.db_path = ATTENDANCE_DIR / "master_attendance.db"
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_id TEXT,
                name TEXT,
                event_type TEXT,
                confidence REAL,
                date TEXT,
                timestamp TEXT,
                snapshot TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_emp_date
            ON attendance_logs (emp_id, date)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_date_time
            ON attendance_logs (date, timestamp)
        """)
        self.conn.commit()

    def _today_str(self):
        return datetime.now().strftime("%Y-%m-%d")

    def _is_on_cooldown(self, emp_id):
        if emp_id not in self.last_seen:
            return False
        return (time.time() - self.last_seen[emp_id]) < self.cooldown_seconds

    def _get_event_type_for_day(self, emp_id, date_str):
        cursor = self.conn.execute("""
            SELECT 1
            FROM attendance_logs
            WHERE emp_id = ? AND date = ?
            LIMIT 1
        """, (emp_id, date_str))
        row = cursor.fetchone()
        return "CHECK_IN" if row is None else "LAST_SEEN"

    def _save_crop(self, frame, bbox, save_path):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        pad = 20
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size > 0:
            cv2.imwrite(str(save_path), crop)

    def _fetch_day_logs_df(self, date_str):
        query = """
            SELECT emp_id, name, event_type, confidence, date, timestamp, snapshot
            FROM attendance_logs
            WHERE date = ?
            ORDER BY timestamp ASC
        """
        return pd.read_sql_query(query, self.conn, params=(date_str,))

    def log_recognition(self, emp_id, name, confidence, frame, bbox):
        if self._is_on_cooldown(emp_id):
            return False, "Cooldown"

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        dt_iso = now.isoformat(timespec="seconds")

        event_type = self._get_event_type_for_day(emp_id, date_str)
        self.last_seen[emp_id] = time.time()

        filename = f"{emp_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        self._save_crop(frame, bbox, SNAPSHOTS_DIR / filename)

        self.conn.execute("""
            INSERT INTO attendance_logs (emp_id, name, event_type, confidence, date, timestamp, snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            emp_id,
            name,
            event_type,
            float(confidence),
            date_str,
            dt_iso,
            filename
        ))
        self.conn.commit()

        self.export_csv(date_str)
        self.export_json(date_str)

        return True, event_type

    def export_csv(self, date_str=None):
        date_str = date_str or self._today_str()
        path = ATTENDANCE_DIR / f"attendance_{date_str}.csv"
        df = self._fetch_day_logs_df(date_str)
        df.to_csv(path, index=False)
        return path

    def export_json(self, date_str=None):
        date_str = date_str or self._today_str()
        path = ATTENDANCE_DIR / f"attendance_{date_str}.json"
        df = self._fetch_day_logs_df(date_str)
        with open(path, "w") as f:
            json.dump(
                {
                    "date": date_str,
                    "events": df.to_dict(orient="records")
                },
                f,
                indent=2
            )
        return path

# ============================================================
# LOAD FACE MODEL
# ============================================================

print("Loading InsightFace model...")
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(512, 512))
print("Model ready")

# ============================================================
# QUALITY HELPERS
# ============================================================

def estimate_sharpness(face_crop):
    if face_crop is None or face_crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def landmarks_quality_ok(face, bbox_small, frame_small):
    x1, y1, x2, y2 = [int(v) for v in bbox_small]
    w, h = max(1, x2 - x1), max(1, y2 - y1)

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False
    if float(getattr(face, "det_score", 1.0)) < MIN_DET_SCORE:
        return False

    crop = frame_small[max(0, y1):min(frame_small.shape[0], y2), max(0, x1):min(frame_small.shape[1], x2)]
    if estimate_sharpness(crop) < MIN_SHARPNESS:
        return False

    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 2:
        return True

    left_eye = np.array(kps[0], dtype=np.float32)
    right_eye = np.array(kps[1], dtype=np.float32)
    if (np.linalg.norm(right_eye - left_eye) / w) < MIN_INTEROCULAR_RATIO:
        return False

    return True

# ============================================================
# EMBEDDINGS & RECOGNITION
# ============================================================

def generate_embeddings():
    print("\n--- Generating New Embeddings ---")
    embeddings = {}
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found: {DATASET_DIR}")
        return embeddings

    for person_folder in DATASET_DIR.iterdir():
        if not person_folder.is_dir():
            continue

        person_name = person_folder.name
        person_embeddings = []

        for img_path in person_folder.glob("*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img, max_num=1)
            if faces:
                person_embeddings.append(faces[0].normed_embedding)

        if person_embeddings:
            mean_emb = np.mean(person_embeddings, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                embeddings[person_name] = (mean_emb / norm).astype(np.float32)
                print(f"✅ Enrolled: {person_name}")

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        return generate_embeddings()
    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)

class FaceRecognitionEngine:
    def __init__(self, database, threshold=0.50):
        self.threshold = threshold
        self.names = list(database.keys())
        self.matrix = (
            np.stack([database[n] for n in self.names]).astype(np.float32)
            if self.names else np.empty((0, 512), dtype=np.float32)
        )

    def identify(self, face_embedding):
        if len(self.names) == 0:
            return None, "Unknown", 0.0

        emb = np.asarray(face_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None, "Unknown", 0.0
        emb = emb / norm

        scores = np.dot(self.matrix, emb)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < self.threshold:
            return None, "Unknown", best_score

        best_name = self.names[best_idx]
        return best_name, best_name, best_score

# ============================================================
# TRACKER COMPATIBILITY
# ============================================================

def create_tracker():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise AttributeError("No supported OpenCV tracker found. Install opencv-contrib-python.")

# ============================================================
# TRACK CLASS
# ============================================================

class Track:
    def __init__(self, track_id, bbox, frame):
        self.track_id = track_id
        self.bbox = bbox
        self.name = "Unknown"
        self.emp_id = None
        self.confidence = 0.0
        self.embedding = None

        self.name_history = deque(maxlen=8)
        self.score_history = deque(maxlen=8)
        self.unknown_run = 0
        self.missed_frames = 0
        self.is_spoof = False
        self.liveness_checked = False
        self.is_live = False
        self.real_count = 0
        self.spoof_count = 0

        self.is_confirmed = False
        self.logged_once = False

        self.tracker = create_tracker()
        x, y, w, h = [int(v) for v in (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])]
        self.tracker.init(frame, (x, y, w, h))

    def reinit(self, frame, bbox):
        self.bbox = bbox
        self.tracker = create_tracker()
        x, y, w, h = [int(v) for v in (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])]
        self.tracker.init(frame, (x, y, w, h))

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if ok:
            self.bbox = [int(v) for v in (box[0], box[1], box[0] + box[2], box[1] + box[3])]
        return ok

# ============================================================
# FAST PIPELINE
# ============================================================

class FastFacePipeline:
    def __init__(self, face_app, recognition_engine, logger, anti_spoof_engine):
        self.face_app = face_app
        self.recognition_engine = recognition_engine
        self.logger = logger
        self.anti_spoof = anti_spoof_engine
        self.frame_index = 0
        self.next_track_id = 1
        self.tracks = {}

    def iou(self, a, b):
        xA, yA, xB, yB = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / float(area_a + area_b - inter)

    def process(self, frame):
        if frame is None:
            return None, None

        self.frame_index += 1

        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / max(1, w)
        frame_small = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))

        lab = cv2.cvtColor(frame_small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        frame_filtered = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        if self.frame_index % DETECT_EVERY_N_FRAMES == 1:
            faces = self.face_app.get(frame_filtered, max_num=MAX_NUM_FACES)
            detections = []

            for f in faces:
                bbox_small = [int(v) for v in f.bbox]
                if not landmarks_quality_ok(f, bbox_small, frame_filtered):
                    continue
                detections.append({
                    "bbox": [int(v / scale) for v in bbox_small],
                    "embedding": f.normed_embedding
                })

            matched_indices = set()

            for tid, tr in list(self.tracks.items()):
                best_iou, best_idx = 0.3, -1

                for i, det in enumerate(detections):
                    if i in matched_indices:
                        continue
                    score = self.iou(det["bbox"], tr.bbox)
                    if score > best_iou:
                        best_iou, best_idx = score, i

                if best_idx != -1:
                    tr.embedding = detections[best_idx]["embedding"]
                    tr.reinit(frame, detections[best_idx]["bbox"])
                    tr.missed_frames = 0
                    matched_indices.add(best_idx)
                else:
                    tr.missed_frames += 1

            for i, det in enumerate(detections):
                if i not in matched_indices:
                    new_tr = Track(self.next_track_id, det["bbox"], frame)
                    new_tr.embedding = det["embedding"]
                    self.tracks[self.next_track_id] = new_tr
                    self.next_track_id += 1

            self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr.missed_frames <= TRACK_TTL}
        else:
            for tid, tr in list(self.tracks.items()):
                if not tr.update(frame):
                    tr.missed_frames += 1

            self.tracks = {tid: tr for tid, tr in self.tracks.items() if tr.missed_frames <= TRACK_TTL}

        if self.frame_index % RECOGNIZE_EVERY_N_FRAMES == 1:
            for tid, tr in self.tracks.items():
                if tr.embedding is None:
                    continue

                emp_id, name, conf = self.recognition_engine.identify(tr.embedding)
                tr.emp_id, tr.name, tr.confidence = emp_id, name, conf

                if name == "Unknown":
                    tr.liveness_checked = False
                    tr.is_spoof = False
                    tr.is_confirmed = False
                    continue

                liveness = self.anti_spoof.is_real(frame, tr.bbox)

                if liveness is None:
                    tr.liveness_checked = False
                    tr.is_spoof = False
                    tr.is_confirmed = False
                    print(f"⚠️ Liveness unavailable for {name}, skipping spoof decision")
                    continue

                tr.liveness_checked = True

                if liveness:
                    tr.is_spoof = False

                    if not tr.logged_once:
                        tr.is_confirmed = True
                        logged, event_type = self.logger.log_recognition(
                            emp_id, name, conf, frame, tr.bbox
                        )
                        if logged:
                            print(f"✅ {event_type}: {name} ({conf:.3f})")
                        tr.logged_once = True
                else:
                    tr.is_spoof = True
                    tr.is_confirmed = False
                    print(f"🚫 Spoof rejected for {name}")

        annotated = self.draw(frame_filtered.copy(), scale)
        return annotated, frame_small

    def draw(self, frame, scale=1.0):
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = [int(v * scale) for v in tr.bbox]

            if tr.is_spoof:
                color = (0, 0, 255)
                label = f"SPOOF DETECTED - {tr.name}"
            elif tr.name != "Unknown":
                color = (0, 255, 0)
                label = f"{tr.name} ({tr.confidence:.2f})"
            else:
                color = (0, 165, 255)
                label = f"Unknown ({tr.confidence:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            status = "SPOOF" if tr.is_spoof else "REAL" if tr.liveness_checked else "UNCHECKED"
            cv2.putText(
                frame,
                f"{status} | ID:{tid}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        return frame

# ============================================================
# RUN
# ============================================================

def start_recognition():
    database = load_embeddings()
    print(f"Loaded embeddings for {len(database)} identities")

    recognition_engine = FaceRecognitionEngine(database, threshold=RECOGNITION_THRESHOLD)
    attendance_logger = AttendanceLogger(cooldown_seconds=LOG_COOLDOWN_SECONDS)
    anti_spoof_engine = AntiSpoofEngine(ANTI_SPOOF_MODEL_DIR, device_id=ANTI_SPOOF_DEVICE_ID)
    

    pipeline = FastFacePipeline(app, recognition_engine, attendance_logger, anti_spoof_engine)

    source_str = str(CAMERA_SOURCE).lower()
    is_video_file = source_str.endswith((".mp4", ".avi", ".mov", ".mkv"))

    if is_video_file:
        print(f"🎬 Starting Video File: {CAMERA_SOURCE}")
        cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not cap.isOpened():
            print(f"❌ Could not open video file: {CAMERA_SOURCE}")
            return

        writer = None
        if SAVE_OUTPUT_VIDEO:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_h = int(height * (TARGET_WIDTH / max(1, width)))
            writer = cv2.VideoWriter(
                str(OUTPUT_VIDEO_PATH),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (TARGET_WIDTH * 2, out_h)
            )

        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ End of video")
                break

            annotated, raw_resized = pipeline.process(frame)

            if annotated is not None:
                if writer is not None:
                    writer.write(annotated)

                print(f"Frame: {pipeline.frame_index}, tracks={len(pipeline.tracks)}")

        cap.release()
        if writer is not None:
            writer.release()
            print(f"💾 Saved processed video: {OUTPUT_VIDEO_PATH}")

    else:
        print(f"🚀 Starting Threaded Stream: {CAMERA_SOURCE}")
        vs = VideoStream(CAMERA_SOURCE).start()
        time.sleep(2.0)

        try:
            while True:
                frame = vs.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                annotated, raw_resized = pipeline.process(frame)
                if annotated is not None:
                    print(f"Frame: {pipeline.frame_index}, tracks={len(pipeline.tracks)}")
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            vs.stop()

def pipeline_frame_generator():
    database = load_embeddings()
    print(f"Loaded embeddings for {len(database)} identities")

    recognition_engine = FaceRecognitionEngine(database, threshold=RECOGNITION_THRESHOLD)
    attendance_logger = AttendanceLogger(cooldown_seconds=LOG_COOLDOWN_SECONDS)
    anti_spoof_engine = AntiSpoofEngine(ANTI_SPOOF_MODEL_DIR, device_id=ANTI_SPOOF_DEVICE_ID)

    pipeline = FastFacePipeline(app, recognition_engine, attendance_logger, anti_spoof_engine)

    source_str = str(CAMERA_SOURCE).lower()
    is_video_file = source_str.endswith((".mp4", ".avi", ".mov", ".mkv"))

    if is_video_file:
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {CAMERA_SOURCE}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, raw_resized = pipeline.process(frame)
            if annotated is not None:
                
                yield annotated

        cap.release()

    else:
        vs = VideoStream(CAMERA_SOURCE).start()
        time.sleep(2.0)

        try:
            while True:
                frame = vs.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                annotated, raw_resized = pipeline.process(frame)
                if annotated is not None:
                    yield annotated
        finally:
            vs.stop()

# if __name__ == "__main__":
#   start_recognition()
