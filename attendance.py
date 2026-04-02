import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from config import ATTENDANCE_DIR, LOG_COOLDOWN_SECONDS, SNAPSHOTS_DIR, SAVE_SNAPSHOTS


class AttendanceLogger:
    def __init__(self, cooldown_seconds=LOG_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.last_seen = {}
        self.db_path = ATTENDANCE_DIR / "master_attendance.db"
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_id TEXT,
                name TEXT,
                event_type TEXT,
                confidence REAL,
                liveness_status TEXT,
                date TEXT,
                timestamp TEXT,
                snapshot TEXT
            )
            """
        )

        self.conn.commit()

    def _today_str(self):
        return datetime.now().strftime("%Y-%m-%d")

    def _is_on_cooldown(self, emp_id):
        if emp_id not in self.last_seen:
            return False
        return (time.time() - self.last_seen[emp_id]) < self.cooldown_seconds

    def log_recognition(
        self,
        emp_id,
        name,
        confidence,
        frame,
        bbox,
        liveness_status="UNKNOWN",
    ):
        if emp_id is None:
            return False, "Invalid"

        if self._is_on_cooldown(emp_id):
            return False, "Cooldown"

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        dt_iso = now.isoformat(timespec="seconds")

        self.last_seen[emp_id] = time.time()

        filename = ""
        if SAVE_SNAPSHOTS:
            filename = f"{emp_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            self._save_crop(frame, bbox, SNAPSHOTS_DIR / filename)

        self.conn.execute(
            """
            INSERT INTO attendance_logs
            (emp_id, name, event_type, confidence, liveness_status, date, timestamp, snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                emp_id,
                name,
                "CHECK_IN",
                float(confidence),
                liveness_status,
                date_str,
                dt_iso,
                filename,
            ),
        )
        self.conn.commit()

        return True, "CHECK_IN"

    def _save_crop(self, frame, bbox, save_path: Path):
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]

        crop = frame[
            max(0, y1):min(h, y2),
            max(0, x1):min(w, x2),
        ]

        if crop.size > 0:
            cv2.imwrite(str(save_path), crop)