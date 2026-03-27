from flask import Flask, Response
import cv2
import threading
import time

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()

def update_stream_from_pipeline():
    global latest_frame

    from main_phase2 import pipeline_frame_generator

    for frame in pipeline_frame_generator():
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        with frame_lock:
            latest_frame = jpeg.tobytes()

def mjpeg_generator():
    global latest_frame
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.01)

@app.route("/")
def index():
    return """
    <html>
      <head><title>Live Detection</title></head>
      <body style="margin:0;background:#111;text-align:center;">
        <h2 style="color:white;">Live AI Detection</h2>
        <img src="/video_feed" style="max-width:95vw;max-height:90vh;border:2px solid #444;" />
      </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    t = threading.Thread(target=update_stream_from_pipeline, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=8081, threaded=True)
    