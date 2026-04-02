import pickle
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from attendance import AttendanceLogger
from config import (
    ANTI_SPOOF_DEVICE_ID,
    ANTI_SPOOF_MODEL_DIR,
    EMBEDDINGS_PATH,
    FACE_CTX_ID,
    FACE_PROVIDERS,
    MAX_NUM_FACES,
    MIN_DET_SCORE,
    MIN_FACE_SIZE,
    MIN_INTEROCULAR_RATIO,
    MIN_SHARPNESS,
    MODEL_DET_SIZE,
    MODEL_NAME,
    RECOGNITION_STABLE_COUNT,
    RECOGNITION_THRESHOLD,
    RECOGNITION_WINDOW,
    STALE_FRAME_MS,
)
import state


# ------------------------------------------------------------
# OPTIONAL ANTI-SPOOF
# ------------------------------------------------------------
class AntiSpoofEngine:
    def __init__(self, model_dir, device_id=-1):
        self.model_dir = str(model_dir)
        self.device_id = device_id
        self.enabled = False

    def is_real(self, frame, bbox):
        return None


# ------------------------------------------------------------
# QUALITY HELPERS
# ------------------------------------------------------------
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

    crop = frame_small[
        max(0, y1):min(frame_small.shape[0], y2),
        max(0, x1):min(frame_small.shape[1], x2),
    ]
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


# ------------------------------------------------------------
# EMBEDDINGS / RECOGNITION
# ------------------------------------------------------------
def build_face_app():
    print("[INFO] Loading InsightFace model...")
    model = FaceAnalysis(name=MODEL_NAME, providers=FACE_PROVIDERS)
    model.prepare(ctx_id=FACE_CTX_ID, det_size=MODEL_DET_SIZE)
    print("[INFO] Model ready")
    return model


def generate_embeddings(face_app, dataset_dir):
    embeddings = {}

    if not dataset_dir.exists():
        return embeddings

    for person_folder in dataset_dir.iterdir():
        if not person_folder.is_dir():
            continue

        person_name = person_folder.name
        person_embeddings = []

        for img_path in person_folder.glob("*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = face_app.get(img, max_num=1)
            if faces:
                person_embeddings.append(faces[0].normed_embedding)

        if person_embeddings:
            mean_emb = np.mean(person_embeddings, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                embeddings[person_name] = (mean_emb / norm).astype(np.float32)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


def load_embeddings(face_app):
    if not EMBEDDINGS_PATH.exists():
        from config import DATASET_DIR
        return generate_embeddings(face_app, DATASET_DIR)

    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        from config import DATASET_DIR
        return generate_embeddings(face_app, DATASET_DIR)


class FaceRecognitionEngine:
    def __init__(self, database, threshold=RECOGNITION_THRESHOLD):
        self.threshold = threshold
        self.names = list(database.keys())
        self.matrix = (
            np.stack([database[n] for n in self.names]).astype(np.float32)
            if self.names
            else np.empty((0, 512), dtype=np.float32)
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


# ------------------------------------------------------------
# SIMPLE TRACK STATE
# ------------------------------------------------------------
class IdentitySmoother:
    def __init__(self, window=RECOGNITION_WINDOW, stable_count=RECOGNITION_STABLE_COUNT):
        self.window = window
        self.stable_count = stable_count
        self.name_history = deque(maxlen=window)
        self.score_history = deque(maxlen=window)
        self.logged_name = None

    def update(self, name, score):
        self.name_history.append(name)
        self.score_history.append(float(score))

    def current_decision(self):
        if not self.name_history:
            return None, "Unknown", 0.0, False

        names = list(self.name_history)
        scores = list(self.score_history)

        majority_name, majority_count = Counter(names).most_common(1)[0]
        median_score = float(np.median(scores)) if scores else 0.0
        stable = majority_name != "Unknown" and majority_count >= self.stable_count

        emp_id = None if majority_name == "Unknown" else majority_name
        return emp_id, majority_name, median_score, stable


# ------------------------------------------------------------
# IOU
# ------------------------------------------------------------
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / float(union)


# ------------------------------------------------------------
# INFERENCE PIPELINE
# ------------------------------------------------------------
class InferencePipeline:
    def __init__(self):
        self.face_app = build_face_app()
        self.database = load_embeddings(self.face_app)
        print(f"[INFO] Loaded embeddings: {len(self.database)} identities")
        print(f"[INFO] Names: {list(self.database.keys())}")
        self.recognition_engine = FaceRecognitionEngine(self.database)
        self.attendance_logger = AttendanceLogger()
        self.anti_spoof = AntiSpoofEngine(ANTI_SPOOF_MODEL_DIR, device_id=ANTI_SPOOF_DEVICE_ID)

        self.smoother = IdentitySmoother()
        self.last_bbox = None

    def process(self, frame):
        if frame is None:
            return None, []

        faces = self.face_app.get(frame, max_num=MAX_NUM_FACES)
        detections = []
        annotated = frame.copy()

        for face in faces:
            bbox = [int(v) for v in face.bbox]
            if not landmarks_quality_ok(face, bbox, frame):
                continue

            emp_id, name, conf = self.recognition_engine.identify(face.normed_embedding)
            detections.append(
                {
                    "bbox": bbox,
                    "emp_id": emp_id,
                    "name": name,
                    "confidence": conf,
                    "embedding": face.normed_embedding,
                }
            )

        if detections:
            det = max(
                detections,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            )

            self.smoother.update(det["name"], det["confidence"])
            smooth_emp_id, smooth_name, smooth_conf, stable = self.smoother.current_decision()

            liveness_status = "UNKNOWN"
            if smooth_name != "Unknown":
                liveness = self.anti_spoof.is_real(frame, det["bbox"])
                if liveness is True:
                    liveness_status = "REAL"
                elif liveness is False:
                    liveness_status = "SPOOF"

            if stable and smooth_name != "Unknown" and self.smoother.logged_name != smooth_name:
                logged, event_type = self.attendance_logger.log_recognition(
                    smooth_emp_id,
                    smooth_name,
                    smooth_conf,
                    frame,
                    det["bbox"],
                    liveness_status=liveness_status,
                )
                if logged:
                    self.smoother.logged_name = smooth_name
                    with state.status_lock:
                        state.latest_events.append(
                            {
                                "name": smooth_name,
                                "emp_id": smooth_emp_id,
                                "confidence": smooth_conf,
                                "event_type": event_type,
                                "liveness_status": liveness_status,
                                "timestamp": time.time(),
                            }
                        )
                        state.latest_events = state.latest_events[-20:]

            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                color = (0, 255, 0) if d["name"] != "Unknown" else (0, 165, 255)
                label = f"{d['name']} ({d['confidence']:.2f})"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

            self.last_bbox = det["bbox"]
        else:
            self.smoother.update("Unknown", 0.0)
            self.last_bbox = None

        return annotated, detections


# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------
def inference_worker():
    print("[INFO] Inference worker started")
    pipeline = InferencePipeline()

    while True:
        try:
            with state.status_lock:
                state.latest_status["inference_running"] = True
                state.latest_status["last_error"] = ""

            with state.raw_lock:
                frame = None if state.latest_raw_frame is None else state.latest_raw_frame.copy()
                raw_ts = state.latest_raw_ts
                
            print("[DEBUG] Processing frame")
            
            if frame is None:
                time.sleep(0.01)
                continue

            frame_age_ms = (time.time() - raw_ts) * 1000.0 if raw_ts else 0.0
            if frame_age_ms > STALE_FRAME_MS:
                time.sleep(0.005)
                continue

            infer_start = time.time()
            annotated, detections = pipeline.process(frame)
            infer_end = time.time()

            if annotated is not None:
                cv2.putText(
                    annotated,
                    f"capture_age={int(frame_age_ms)}ms infer={int((infer_end-infer_start)*1000)}ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                with state.result_lock:
                    state.latest_result_frame = annotated
                    state.latest_result_ts = infer_end
                    state.latest_detections = detections

                with state.status_lock:
                    state.latest_status["last_inference_ts"] = infer_end

        except Exception as e:
            with state.status_lock:
                state.latest_status["last_error"] = f"inference_worker: {e}"
            time.sleep(0.5)


def ensure_inference_started():
    with state.startup_lock:
        if state.inference_started:
            return

        t = threading.Thread(target=inference_worker, daemon=True)
        t.start()
        state.inference_started = True