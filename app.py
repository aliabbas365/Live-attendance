from flask import Flask, Response, jsonify
import cv2
import time

import state
from config import APP_HOST, APP_PORT, MJPEG_SLEEP_SECONDS
from capture_service import ensure_capture_started
from inference_service import ensure_inference_started


app = Flask(__name__)


# ------------------------------------------------------------
# START PIPELINE (ONLY ONCE)
# ------------------------------------------------------------
def ensure_pipeline_started():
    ensure_capture_started()
    ensure_inference_started()


# ------------------------------------------------------------
# MJPEG STREAM (DEBUG ONLY)
# ------------------------------------------------------------
def mjpeg_generator():
    last_frame = None

    while True:
        with state.result_lock:
            frame = state.latest_result_frame

        if frame is None:
            time.sleep(0.01)
            continue

        # encode to jpeg
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()

        # skip duplicate frames
        if frame_bytes == last_frame:
            time.sleep(MJPEG_SLEEP_SECONDS)
            continue

        last_frame = frame_bytes

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.route("/")
def index():
    ensure_pipeline_started()

    return """
    <html>
      <head><title>AI Face Recognition</title></head>
      <body style="margin:0;background:#111;text-align:center;">
        <h2 style="color:white;">Live AI Detection (Production Pipeline)</h2>
        <img src="/video_feed" style="max-width:95vw;max-height:90vh;border:2px solid #444;" />
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    ensure_pipeline_started()
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/health")
def health():
    with state.status_lock:
        status = dict(state.latest_status)

    return jsonify({
        "status": status,
        "events_count": len(state.latest_events),
    })


@app.route("/latest_events")
def latest_events():
    with state.status_lock:
        events = list(state.latest_events)

    return jsonify(events)


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    ensure_pipeline_started()
    app.run(host=APP_HOST, port=APP_PORT, threaded=True)