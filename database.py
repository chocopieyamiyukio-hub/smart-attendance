"""SQLite persistence for students and session-based attendance."""

from __future__ import annotations

import csv
import gc
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime
from datetime import time as clock_time
from pathlib import Path

from config import DATABASE_PATH, EXPORTS_DIR, ensure_runtime_directories

_ACTIVE_CONNECTIONS: set[sqlite3.Connection] = set()

# A student may be present once in each scheduled teaching session. Dashboard
# totals therefore count every saved attendance record, including sessions.
ATTENDANCE_SESSIONS = (
    ("8:45to10:45", clock_time(8, 45), clock_time(10, 45)),
    ("11:00to12:00", clock_time(11, 0), clock_time(12, 0)),
    ("13:00to14:00", clock_time(13, 0), clock_time(14, 0)),
    ("14:15to16:15", clock_time(14, 15), clock_time(16, 15)),
)


def attendance_session(at: datetime | None = None) -> str | None:
    """Return the configured attendance session for *at*, or None outside it."""
    current_time = (at or datetime.now()).time()
    for label, start, end in ATTENDANCE_SESSIONS:
        if start <= current_time <= end:
            return label
    return None


def _connection() -> sqlite3.Connection:
    ensure_runtime_directories()
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 3000")
    _ACTIVE_CONNECTIONS.add(connection)
    return connection


@contextmanager
def _managed_connection():
    connection = _connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
        _ACTIVE_CONNECTIONS.discard(connection)
        gc.collect()


def close_all_connections() -> None:
    for connection in list(_ACTIVE_CONNECTIONS):
        try:
            connection.close()
        except Exception:
            pass
        _ACTIVE_CONNECTIONS.discard(connection)
    gc.collect()


def init_db() -> None:
    """Initialise the database tables if they do not exist."""
    with _managed_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                session TEXT NOT NULL DEFAULT 'Legacy',
                FOREIGN KEY(student_id) REFERENCES Students(student_id)
            )
            """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(Attendance)")}
        if "session" not in columns:
            conn.execute(
                "ALTER TABLE Attendance ADD COLUMN session TEXT NOT NULL DEFAULT 'Legacy'"
            )
        # Replace the old daily-only key so the same student can attend a
        # later scheduled session, but never twice in the same session.
        conn.execute("DROP INDEX IF EXISTS idx_daily_attendance")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_session_attendance "
            "ON Attendance(student_id, date, session)"
        )


def add_student(student_id: str, name: str) -> None:
    """Create or update a student record."""
    cleaned_student_id = student_id.strip()
    cleaned_name = name.strip()
    if not cleaned_student_id or not cleaned_name:
        raise ValueError("Student ID and name are required.")
    with _managed_connection() as conn:
        conn.execute(
            "INSERT INTO Students (student_id, name) VALUES (?, ?) "
            "ON CONFLICT(student_id) DO UPDATE SET name = excluded.name",
            (cleaned_student_id, cleaned_name),
        )


def update_student(old_student_id: str, student_id: str, name: str) -> None:
    """Update a student's ID and name, preserving all attendance history."""
    old_id, new_id, cleaned_name = (
        old_student_id.strip(),
        student_id.strip(),
        name.strip(),
    )
    if not old_id or not new_id or not cleaned_name:
        raise ValueError("Student ID and name are required.")
    with _managed_connection() as conn:
        if (
            conn.execute(
                "SELECT 1 FROM Students WHERE student_id = ?", (old_id,)
            ).fetchone()
            is None
        ):
            raise ValueError("The selected student no longer exists.")
        if (
            old_id != new_id
            and conn.execute(
                "SELECT 1 FROM Students WHERE student_id = ?", (new_id,)
            ).fetchone()
            is not None
        ):
            raise ValueError("That Student ID is already registered.")
        if old_id != new_id:
            # Add the replacement first so the attendance foreign key remains
            # valid while its references are moved inside this transaction.
            conn.execute(
                "INSERT INTO Students (student_id, name) VALUES (?, ?)",
                (new_id, cleaned_name),
            )
            conn.execute(
                "UPDATE Attendance SET student_id = ? WHERE student_id = ?",
                (new_id, old_id),
            )
            conn.execute("DELETE FROM Students WHERE student_id = ?", (old_id,))
        else:
            conn.execute(
                "UPDATE Students SET name = ? WHERE student_id = ?",
                (cleaned_name, old_id),
            )


def delete_student(student_id: str) -> bool:
    """Delete one student and every attendance record belonging to them."""
    cleaned_student_id = student_id.strip()
    with _managed_connection() as conn:
        conn.execute(
            "DELETE FROM Attendance WHERE student_id = ?", (cleaned_student_id,)
        )
        cursor = conn.execute(
            "DELETE FROM Students WHERE student_id = ?", (cleaned_student_id,)
        )
        return cursor.rowcount == 1


def delete_all_students_and_attendance() -> None:
    """Clear database registration data without deleting the database file."""
    with _managed_connection() as conn:
        conn.execute("DELETE FROM Attendance")
        conn.execute("DELETE FROM Students")


def delete_today_attendance() -> None:
    with _managed_connection() as conn:
        conn.execute(
            "DELETE FROM Attendance WHERE date = ?",
            (datetime.now().strftime("%Y-%m-%d"),),
        )

def student_exists(student_id: str) -> bool:
    with _managed_connection() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM Students WHERE student_id = ?",
                (student_id.strip(),),
            ).fetchone()
            is not None
        )


def get_student(student_id: str) -> tuple[str, str] | None:
    with _managed_connection() as conn:
        return conn.execute(
            "SELECT student_id, name FROM Students WHERE student_id = ?",
            (student_id.strip(),),
        ).fetchone()


def list_students() -> list[tuple[str, str]]:
    """Return every registered student ordered by student ID."""
    with _managed_connection() as conn:
        return conn.execute(
            "SELECT student_id, name FROM Students ORDER BY student_id"
        ).fetchall()


def dashboard_summary() -> dict[str, int]:
    """Return lightweight metrics for the reports dashboard."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _managed_connection() as conn:
        return {
            "students": conn.execute("SELECT COUNT(*) FROM Students").fetchone()[0],
            "today": conn.execute(
                "SELECT COUNT(DISTINCT student_id) FROM Attendance WHERE date = ?",
                (today,),
            ).fetchone()[0],
            "records": conn.execute(
                "SELECT COUNT(*) FROM Attendance WHERE date = ?", (today,)
            ).fetchone()[0],
        }


def recent_attendance(limit: int = 12) -> list[tuple[str, str, str, str]]:
    with _managed_connection() as conn:
        return conn.execute(
            "SELECT a.time, a.session, s.student_id, s.name FROM Attendance a "
            "JOIN Students s ON s.student_id = a.student_id "
            "WHERE a.date = ? ORDER BY a.id DESC LIMIT ?",
            (datetime.now().strftime("%Y-%m-%d"), limit),
        ).fetchall()


def mark_attendance(student_id: str, at: datetime | None = None) -> bool:
    """Record a student once per date/session; ignore scans outside sessions."""
    cleaned_student_id = student_id.strip()
    if not cleaned_student_id:
        return False

    now = at or datetime.now()
    session = attendance_session(now)
    if session is None:
        return False
    with _managed_connection() as conn:
        student = conn.execute(
            "SELECT 1 FROM Students WHERE student_id = ?",
            (cleaned_student_id,),
        ).fetchone()
        if student is None:
            return False

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO Attendance
            (student_id, date, time, session)
            VALUES (?, ?, ?, ?)
            """,
            (
                cleaned_student_id,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                session,
            ),
        )
        inserted = conn.execute("SELECT changes()").fetchone()[0]
        return inserted == 1


def export_to_csv(
    output_path: Path | None = None, report_date: str | None = None
) -> Path:
    """Export only one day's attendance; today is used unless specified."""
    ensure_runtime_directories()
    destination = output_path or EXPORTS_DIR / "attendance_report.csv"
    selected_date = report_date or datetime.now().strftime("%Y-%m-%d")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _managed_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.date, a.time, a.session, s.student_id, s.name
            FROM Attendance a
            JOIN Students s ON a.student_id = s.student_id
            WHERE a.date = ?
            ORDER BY a.date DESC, a.time DESC
            """,
            (selected_date,),
        ).fetchall()
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("date", "time", "session", "student_id", "name"))
        writer.writerows(rows)
    return destination
