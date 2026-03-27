import sqlite3
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "attendance_logs" / "master_attendance.db"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"

# Usage:
#   python report_generator.py
#   python report_generator.py 2026-03-26
TARGET_DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")


def snapshot_path(filename):
    if not filename:
        return ""
    return str(SNAPSHOTS_DIR / filename)


def generate_report():
    if not DB_PATH.exists():
        print(f"No master database found at {DB_PATH}. Run your main app first!")
        return

    conn = sqlite3.connect(DB_PATH)

    print(f"\n{'=' * 70}")
    print(f"ATTENDANCE REPORT FOR {TARGET_DATE}")
    print(f"{'=' * 70}\n")

    # ---------------------------------------------------------
    # QUERY 1: Raw Event Log for the Target Date
    # ---------------------------------------------------------
    print("1. FULL EVENT LOG:\n")

    query_raw = """
        SELECT
            emp_id,
            name,
            event_type,
            timestamp,
            confidence,
            snapshot
        FROM attendance_logs
        WHERE date = ?
        ORDER BY timestamp ASC
    """

    df_raw = pd.read_sql_query(query_raw, conn, params=(TARGET_DATE,))

    if df_raw.empty:
        print(f"No records found for {TARGET_DATE}.")
        conn.close()
        return

    df_raw["snapshot_path"] = df_raw["snapshot"].apply(snapshot_path)

    print(df_raw.to_string(index=False))
    print("\n" + "-" * 70 + "\n")

    # ---------------------------------------------------------
    # QUERY 2: Daily Summary
    # ---------------------------------------------------------
    print("2. DAILY EMPLOYEE SUMMARY:\n")

    query_summary = """
        SELECT
            a1.emp_id,
            a1.name,
            MIN(a1.timestamp) AS first_in,
            MAX(a1.timestamp) AS last_seen,
            COUNT(*) AS total_scans,
            (
                SELECT a2.snapshot
                FROM attendance_logs a2
                WHERE a2.date = ? AND a2.emp_id = a1.emp_id
                ORDER BY a2.timestamp ASC
                LIMIT 1
            ) AS first_snapshot,
            (
                SELECT a3.snapshot
                FROM attendance_logs a3
                WHERE a3.date = ? AND a3.emp_id = a1.emp_id
                ORDER BY a3.timestamp DESC
                LIMIT 1
            ) AS last_snapshot
        FROM attendance_logs a1
        WHERE a1.date = ?
        GROUP BY a1.emp_id, a1.name
        ORDER BY first_in ASC
    """

    df_summary = pd.read_sql_query(
        query_summary,
        conn,
        params=(TARGET_DATE, TARGET_DATE, TARGET_DATE)
    )

    df_summary["first_in_dt"] = pd.to_datetime(df_summary["first_in"])
    df_summary["last_seen_dt"] = pd.to_datetime(df_summary["last_seen"])

    def calculate_hours(row):
        if row["first_in_dt"] != row["last_seen_dt"] and row["total_scans"] > 1:
            delta = row["last_seen_dt"] - row["first_in_dt"]
            return round(delta.total_seconds() / 3600, 2)
        return 0.0

    df_summary["hours_logged"] = df_summary.apply(calculate_hours, axis=1)

    df_summary["first_in"] = df_summary["first_in_dt"].dt.strftime("%H:%M:%S")
    df_summary["last_seen"] = df_summary["last_seen_dt"].dt.strftime("%H:%M:%S")

    df_summary["first_snapshot_path"] = df_summary["first_snapshot"].apply(snapshot_path)
    df_summary["last_snapshot_path"] = df_summary["last_snapshot"].apply(snapshot_path)

    print(
        df_summary[
            [
                "emp_id",
                "name",
                "first_in",
                "last_seen",
                "hours_logged",
                "total_scans",
                "first_snapshot",
                "last_snapshot",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 70 + "\n")

    # ---------------------------------------------------------
    # QUERY 3: CHECK-IN ONLY VIEW
    # ---------------------------------------------------------
    print("3. CHECK-IN EVENTS ONLY:\n")

    query_checkins = """
        SELECT
            emp_id,
            name,
            timestamp,
            confidence,
            snapshot
        FROM attendance_logs
        WHERE date = ?
          AND event_type = 'CHECK_IN'
        ORDER BY timestamp ASC
    """

    df_checkins = pd.read_sql_query(query_checkins, conn, params=(TARGET_DATE,))
    df_checkins["snapshot_path"] = df_checkins["snapshot"].apply(snapshot_path)

    if df_checkins.empty:
        print("No CHECK_IN records found.")
    else:
        print(df_checkins.to_string(index=False))

    print("\n" + "=" * 70 + "\n")
    conn.close()


if __name__ == "__main__":
    generate_report()