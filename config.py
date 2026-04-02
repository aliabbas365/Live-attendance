import os
from pathlib import Path

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
except Exception:
    torch = None


# ------------------------------------------------------------
# BASE PATHS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "employees_structured"
EMBEDDINGS_PATH = BASE_DIR / "embeddings" / "face_embeddings.pkl"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
ATTENDANCE_DIR = BASE_DIR / "attendance_logs"
ANTI_SPOOF_MODEL_DIR = BASE_DIR / "resources" / "anti_spoof_models"

os.makedirs(BASE_DIR / "embeddings", exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ------------------------------------------------------------
# CAMERA / RTSP
# ------------------------------------------------------------
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "").strip()

if not CAMERA_SOURCE:
    raise RuntimeError(
        "CAMERA_SOURCE environment variable is not set.\n"
        "Example:\n"
        "export CAMERA_SOURCE='rtsp://user:pass@host:port/Streaming/Channels/1001'"
    )

RTSP_OPTIONS = {
    "rtsp_transport": "tcp",
    "fflags": "nobuffer",
    "flags": "low_delay",
    "max_delay": "500000",
    "stimeout": "5000000",
}

CAPTURE_RECONNECT_SECONDS = 2
STALE_FRAME_MS = 250


# ------------------------------------------------------------
# MODEL / GPU
# ------------------------------------------------------------
if ort is not None:
    try:
        ORT_AVAILABLE_PROVIDERS = ort.get_available_providers()
    except Exception:
        ORT_AVAILABLE_PROVIDERS = []
else:
    ORT_AVAILABLE_PROVIDERS = []

USE_ORT_CUDA = "CUDAExecutionProvider" in ORT_AVAILABLE_PROVIDERS
USE_TORCH_CUDA = bool(torch is not None and torch.cuda.is_available())

FACE_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if USE_ORT_CUDA
    else ["CPUExecutionProvider"]
)

FACE_CTX_ID = 0 if USE_ORT_CUDA else -1
ANTI_SPOOF_DEVICE_ID = 0 if USE_TORCH_CUDA else -1

MODEL_NAME = "buffalo_s"
MODEL_DET_SIZE = (512, 512)


# ------------------------------------------------------------
# PIPELINE SETTINGS
# ------------------------------------------------------------
TARGET_WIDTH = 640
MAX_NUM_FACES = 1

RECOGNITION_THRESHOLD = 0.38
LOG_COOLDOWN_SECONDS = 300

MIN_FACE_SIZE = 12
MIN_DET_SCORE = 0.25
MIN_SHARPNESS = 1.0
MIN_INTEROCULAR_RATIO = 0.18

RECOGNITION_STABLE_COUNT = 3
RECOGNITION_WINDOW = 5


# ------------------------------------------------------------
# ATTENDANCE / OUTPUT
# ------------------------------------------------------------
SAVE_SNAPSHOTS = True
SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = BASE_DIR / "output_phase2.mp4"


# ------------------------------------------------------------
# WEB / DEBUG
# ------------------------------------------------------------
APP_HOST = "0.0.0.0"
APP_PORT = 8081
MJPEG_SLEEP_SECONDS = 0.001