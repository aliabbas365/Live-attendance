import time

import av
import cv2
import numpy as np

from config import (
    CAMERA_SOURCE,
    CAPTURE_RECONNECT_SECONDS,
    RTSP_OPTIONS,
    STALE_FRAME_MS,
    TARGET_WIDTH,
)
import state


def _resize_for_pipeline(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame

    if w == TARGET_WIDTH:
        return frame

    scale = TARGET_WIDTH / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (TARGET_WIDTH, new_h))


def _open_container():
    return av.open(
        CAMERA_SOURCE,
        mode="r",
        options=RTSP_OPTIONS,
    )


def capture_worker():
    while True:
        container = None
        try:
            with state.status_lock:
                state.latest_status["capture_running"] = True
                state.latest_status["last_error"] = ""

            container = _open_container()
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"

            for packet in container.demux(video_stream):
                if packet is None:
                    continue

                for frame in packet.decode():
                    now_ts = time.time()

                    img = frame.to_ndarray(format="bgr24")
                    img = _resize_for_pipeline(img)

                    with state.raw_lock:
                        state.latest_raw_frame = img
                        state.latest_raw_ts = now_ts

                    with state.status_lock:
                        state.latest_status["last_capture_ts"] = now_ts

                    age_ms = (time.time() - now_ts) * 1000.0
                    if age_ms > STALE_FRAME_MS:
                        continue

                if not state.latest_status["capture_running"]:
                    break

        except Exception as e:
            with state.status_lock:
                state.latest_status["last_error"] = f"capture_worker: {e}"
            time.sleep(CAPTURE_RECONNECT_SECONDS)

        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass


def ensure_capture_started():
    import threading
    import state

    with state.startup_lock:
        if state.capture_started:
            return

        t = threading.Thread(target=capture_worker, daemon=True)
        t.start()
        state.capture_started = True