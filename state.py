import threading


# ------------------------------------------------------------
# SHARED FRAMES
# ------------------------------------------------------------
latest_raw_frame = None
latest_raw_ts = 0.0

latest_result_frame = None
latest_result_ts = 0.0


# ------------------------------------------------------------
# SHARED METADATA
# ------------------------------------------------------------
latest_detections = []
latest_events = []
latest_status = {
    "capture_running": False,
    "inference_running": False,
    "last_capture_ts": 0.0,
    "last_inference_ts": 0.0,
    "last_error": "",
}


# ------------------------------------------------------------
# LOCKS
# ------------------------------------------------------------
raw_lock = threading.Lock()
result_lock = threading.Lock()
status_lock = threading.Lock()


# ------------------------------------------------------------
# STARTUP FLAGS
# ------------------------------------------------------------
capture_started = False
inference_started = False
startup_lock = threading.Lock()