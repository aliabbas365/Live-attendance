import cv2
import os
import threading

# FIXED: This must be at the very top to prevent the OpenMP #15 crash!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from collections import deque, Counter
from insightface.app import FaceAnalysis

# Suppress PyTorch warnings for a clean terminal output
warnings.filterwarnings("ignore")


# ============================================================
# NEW: THREADED CCTV STREAM READER
# ============================================================
class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force small buffer
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

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

os.makedirs(BASE_DIR / "embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Recognition and Quality Thresholds
RECOGNITION_THRESHOLD = 0.38
MIN_FACE_SIZE = 15       
MIN_DET_SCORE = 0.40     
MIN_SHARPNESS = 2.0     
MIN_INTEROCULAR_RATIO = 0.35

# Temporal stability
STABLE_FRAMES_REQUIRED = 2
UNKNOWN_HOLDOFF_FRAMES = 3
LOG_COOLDOWN_SECONDS = 300 

# Speed tuning for CPU
TARGET_WIDTH = 1280
DETECT_EVERY_N_FRAMES = 3
RECOGNIZE_EVERY_N_FRAMES = 5
MAX_NUM_FACES = 5
TRACK_TTL = 8

# ✅ CCTV RTSP LINK CONFIGURATION
# Format: "rtsp://username:password@IP_ADDRESS:PORT/stream_path"
CAMERA_SOURCE = "/FACE-RECOGNITION/test_video.mp4" 


DRAW_FPS = True
DRAW_TRACK_ID = False
DRAW_ATTENDANCE_PANEL = True

# ============================================================
# SILENT FACE ANTI-SPOOFING ENGINE
# ============================================================

class AntiSpoofEngine:
    def __init__(self, model_dir, device_id=0):
        self.enabled = False
        self.device_id = device_id

        try:
            from src.anti_spoof_predict import AntiSpoofPredict
            from src.generate_patches import CropImage

            self.model_test = AntiSpoofPredict(device_id=0)
            self.image_cropper = CropImage()
            self.model_dir = str(model_dir)
            self.enabled = True

            mode = f"GPU cuda:{device_id}" if device_id >= 0 else "CPU"
            print("🛡️  Silent-Face-Anti-Spoofing loaded successfully! (GPU cuda:0)")

        except ImportError:
            print("❌ WARNING: 'src' folder not found. Liveness checking is DISABLED.")
        except Exception as e:
            print(f"❌ Failed to load Anti-Spoofing models: {e}")

    def is_real(self, frame, bbox):
        if not self.enabled:
            return True 
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        image_bbox = [x1, y1, w, h]
        prediction = np.zeros((1, 3))
        
        try:
            from src.anti_spoof_predict import parse_model_name
            for model_name in os.listdir(self.model_dir):
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
                prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))
            
            label = np.argmax(prediction)
            return label == 1
            
        except Exception as e:
            print(f"Liveness Check Error: {e}")
            return False

# ============================================================
# ATTENDANCE LOGGER
# ============================================================

class AttendanceLogger:
    def __init__(self, cooldown_seconds=300):
        self.cooldown_seconds = cooldown_seconds
        self.last_seen = {}
        self.daily_records = {}
        self.all_logs = []
        self.today = datetime.now().strftime("%Y-%m-%d")
        
        self.db_path = ATTENDANCE_DIR / "master_attendance.db"
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''
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
        ''')
        self.conn.commit()

    def _is_on_cooldown(self, emp_id):
        if emp_id not in self.last_seen:
            return False
        return (time.time() - self.last_seen[emp_id]) < self.cooldown_seconds

    def _save_crop(self, frame, bbox, save_path):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        pad = 20
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size > 0:
            cv2.imwrite(str(save_path), crop)

    def log_recognition(self, emp_id, name, confidence, frame, bbox):
        if self._is_on_cooldown(emp_id):
            return False, "Cooldown"

        now_ts = time.time()
        self.last_seen[emp_id] = now_ts
        dt_iso = datetime.fromtimestamp(now_ts).isoformat()
        
        if emp_id not in self.daily_records:
            event_type = "CHECK_IN"
            self.daily_records[emp_id] = {
                "name": name,
                "check_in": dt_iso,
                "last_seen": dt_iso,
                "status": "ARRIVED",
                "detections": 1
            }
        else:
            event_type = "LAST_SEEN"
            self.daily_records[emp_id]["last_seen"] = dt_iso
            self.daily_records[emp_id]["status"] = "ACTIVE"
            self.daily_records[emp_id]["detections"] += 1

        filename = f"{emp_id}_{datetime.fromtimestamp(now_ts).strftime('%Y%m%d_%H%M%S')}.jpg"
        self._save_crop(frame, bbox, SNAPSHOTS_DIR / filename)

        self.all_logs.append({
            "emp_id": emp_id, "name": name, "event_type": event_type,
            "confidence": round(float(confidence), 4), "timestamp": dt_iso,
            "snapshot": filename
        })
        
        self.conn.execute('''
            INSERT INTO attendance_logs (emp_id, name, event_type, confidence, date, timestamp, snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (emp_id, name, event_type, float(confidence), self.today, dt_iso, filename))
        self.conn.commit()

        self.export_csv()
        return True, event_type

    def export_csv(self):
        path = ATTENDANCE_DIR / f"attendance_{self.today}.csv"
        pd.DataFrame(self.all_logs).to_csv(path, index=False)
        return path

    def export_json(self):
        path = ATTENDANCE_DIR / f"attendance_{self.today}.json"
        with open(path, "w") as f:
            json.dump({"date": self.today, "events": self.all_logs}, f, indent=2)
        return path

# ============================================================
# LOAD FACE MODEL
# ============================================================

from insightface.app import FaceAnalysis

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
    if face_crop is None or face_crop.size == 0: return 0.0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def landmarks_quality_ok(face, bbox_small, frame_small):
    x1, y1, x2, y2 = [int(v) for v in bbox_small]
    w, h = max(1, x2 - x1), max(1, y2 - y1)

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE: return False
    if float(getattr(face, "det_score", 1.0)) < MIN_DET_SCORE: return False

    crop = frame_small[max(0, y1):min(frame_small.shape[0], y2), max(0, x1):min(frame_small.shape[1], x2)]
    if estimate_sharpness(crop) < MIN_SHARPNESS: return False

    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 2: return True

    left_eye, right_eye = np.array(kps[0], dtype=np.float32), np.array(kps[1], dtype=np.float32)
    if (np.linalg.norm(right_eye - left_eye) / w) < MIN_INTEROCULAR_RATIO: return False

    return True

# ============================================================
# EMBEDDINGS & RECOGNITION
# ============================================================

def generate_embeddings():
    print("\n--- Generating New Embeddings ---")
    embeddings = {}
    if not DATASET_DIR.exists():
        return embeddings

    for person_folder in DATASET_DIR.iterdir():
        if not person_folder.is_dir(): continue
        person_name = person_folder.name
        person_embeddings = []

        for img_path in person_folder.glob("*"):
            img = cv2.imread(str(img_path))
            if img is None: continue
            faces = app.get(img, max_num=1)
            if faces: person_embeddings.append(faces[0].normed_embedding)

        if person_embeddings:
            mean_emb = np.mean(person_embeddings, axis=0)
            embeddings[person_name] = mean_emb / np.linalg.norm(mean_emb)
            print(f"✅ Enrolled: {person_name}")

    with open(EMBEDDINGS_PATH, "wb") as f: pickle.dump(embeddings, f)
    return embeddings

def load_embeddings():
    if not EMBEDDINGS_PATH.exists(): return generate_embeddings()
    with open(EMBEDDINGS_PATH, "rb") as f: return pickle.load(f)

class FaceRecognitionEngine:
    def __init__(self, database, threshold=0.50):
        self.threshold = threshold
        self.names = list(database.keys())
        self.matrix = np.stack([database[n] for n in self.names]).astype(np.float32) if self.names else np.array([])

    def identify(self, face_embedding):
        if len(self.names) == 0: return None, "Unknown", 0.0
        scores = np.dot(self.matrix, face_embedding)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < self.threshold: return None, "Unknown", float(scores[best_idx])
        return self.names[best_idx], self.names[best_idx], float(scores[best_idx])

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
        
        # Voting counters for Liveness
        self.is_live = False
        self.real_count = 0
        self.spoof_count = 0
        
        self.is_confirmed = False
        self.logged_once = False
        
     def create_kcf_tracker():
        if hasattr(cv2, "TrackerKCF_create"):
          return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
          return cv2.legacy.TrackerKCF_create()
        raise AttributeError("KCF tracker is not available in this OpenCV build. Install opencv-con trib-python.")  

        self.tracker = create_kcf_tracker()
        
        x, y, w, h = [int(v) for v in (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])]
        self.tracker.init(frame, (x, y, w, h))

    def reinit(self, frame, bbox):
        self.bbox = bbox
        self.tracker = cv2.TrackerKCF_create()
        x, y, w, h = [int(v) for v in (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])]
        self.tracker.init(frame, (x, y, w, h))

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if ok: self.bbox = [int(v) for v in (box[0], box[1], box[0]+box[2], box[1]+box[3])]
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
        return inter / float(max(1, (a[2]-a[0])*(a[3]-a[1])) + max(1, (b[2]-b[0])*(b[3]-b[1])) - inter)

    def process(self, frame):
        if frame is None: return None, None
        self.frame_index += 1
        
        # 1. Standard Resize
        h, w = frame.shape[:2]
        scale = 1280 / w 
        frame_small = cv2.resize(frame, (1280, int(h * scale)))

        # 2. Apply CLAHE Filter (Fixes Lighting)
        lab = cv2.cvtColor(frame_small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Lowered limit for natural look
        cl = clahe.apply(l)
        frame_filtered = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

        # 3. Detection (Use the FILTERED frame)
        if self.frame_index % DETECT_EVERY_N_FRAMES == 1:
            faces = self.face_app.get(frame_filtered, max_num=MAX_NUM_FACES)
            detections = []
            for f in faces:
                bbox_small = [int(v) for v in f.bbox]
                # Map coordinates back to original frame size
                detections.append({
                    "bbox": [int(v / scale) for v in bbox_small],
                    "embedding": f.normed_embedding
                })

            matched_indices = set()
            for tid, tr in self.tracks.items():
                best_iou, best_idx = 0.3, -1
                for i, det in enumerate(detections):
                    if i in matched_indices: continue
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
                if not tr.update(frame): tr.missed_frames += 1

        # Recognition & Log Logic
        if self.frame_index % RECOGNIZE_EVERY_N_FRAMES == 1:
            for tid, tr in self.tracks.items():
                if tr.embedding is not None:
                    emp_id, name, conf = self.recognition_engine.identify(tr.embedding)
                    tr.name, tr.confidence = name, conf
                    
                    if name != "Unknown" and not tr.logged_once:
                        is_real = self.anti_spoof.is_real(frame, tr.bbox)

                        if is_real:
                            tr.is_confirmed = True
                            self.logger.log_recognition(emp_id, name, conf, frame, tr.bbox)
                            tr.logged_once = True
                        else:
                            print(f"Spoof rejected for {name}")

        # 4. Draw on the Filtered Frame so we see the AI's "Vision"
        annotated = self.draw(frame_filtered.copy(), scale)
        
        return annotated, frame_small # Return both for the split screen

    def draw(self, frame, scale=1.0):
        for tid, tr in self.tracks.items():
            x1, y1, x2, y2 = [int(v * scale) for v in tr.bbox]
            # Green if known, Red if Unknown
            color = (0, 255, 0) if tr.name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{tr.name} ({tr.confidence:.2f})"
            cv2.putText(frame, label, (x1, int(y1 - 10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# ============================================================
# RUN (With Auto-Reconnection for CCTV)
# ============================================================

def start_recognition():
    database = load_embeddings()
    recognition_engine = FaceRecognitionEngine(database, threshold=RECOGNITION_THRESHOLD)
    attendance_logger = AttendanceLogger(cooldown_seconds=LOG_COOLDOWN_SECONDS)
    anti_spoof_engine = AntiSpoofEngine(ANTI_SPOOF_MODEL_DIR)

    pipeline = FastFacePipeline(app, recognition_engine, attendance_logger, anti_spoof_engine)

    source_str = str(CAMERA_SOURCE).lower()
    is_video_file = source_str.endswith((".mp4", ".avi", ".mov", ".mkv"))

    if is_video_file:
        print(f"🎬 Starting Video File: {CAMERA_SOURCE}")
        cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not cap.isOpened():
            print(f"❌ Could not open video file: {CAMERA_SOURCE}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(BASE_DIR / "output_test.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (TARGET_WIDTH * 2, int(height * (TARGET_WIDTH / width)))
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ End of video")
                break

            annotated, raw_resized = pipeline.process(frame)

            if annotated is not None:
                split_screen = np.hstack((raw_resized, annotated))
                cv2.putText(split_screen, "RAW VIDEO", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(split_screen, "FILTERED (AI VIEW)", (900, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                writer.write(split_screen)
                print(f"processed frame {pipeline.frame_index}, tracks={len(pipeline.tracks)}")

        writer.release()
        cap.release()

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
                    split_screen = np.hstack((raw_resized, annotated))
                    cv2.putText(split_screen, "RAW CCTV", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(split_screen, "FILTERED (AI VIEW)", (900, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    print(f"processed frame {pipeline.frame_index}, tracks={len(pipeline.tracks)}")

        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            vs.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_recognition()

