"""Central configuration and filesystem helpers for the application."""

from __future__ import annotations

import json
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
EXPORTS_DIR = BASE_DIR / "exports"
LOGS_DIR = BASE_DIR / "logs"
DATABASE_PATH = DATABASE_DIR / "attendance.db"
MODEL_PATH = MODELS_DIR / "face_recognizer.yml"
LABELS_PATH = MODELS_DIR / "labels.json"
SETTINGS_PATH = BASE_DIR / "settings.json"
RECOGNITION_LOG_PATH = EXPORTS_DIR / "recognition_log.csv"
LOG_FILE_PATH = LOGS_DIR / "app.log"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CAMERA_INDEX = 0
FACE_SIZE = (200, 200)
# Face detection tuning for stable live capture and dataset processing.
FACE_DETECTION_SCALE_FACTOR = 1.08
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (100, 100)
# A lower LBPH distance is a closer match. These are safe demonstration
# defaults and can be adjusted in settings.json without changing source code.
RECOGNITION_CONFIDENCE_THRESHOLD = 70.0
CAPTURE_SAMPLE_COUNT = 150
RECOGNITION_COOLDOWN_SECONDS = 15
LBPH_RADIUS = 2
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
MIN_FACE_SHARPNESS = 55.0


def ensure_runtime_directories() -> None:
    """Create all directories produced by the application."""
    for directory in (DATASET_DIR, MODELS_DIR, DATABASE_DIR, EXPORTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    """Configure consistent application logging for console and file output."""
    ensure_runtime_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        ],
        force=True,
    )


def get_settings() -> dict:
    """Return user-editable application settings, creating safe defaults."""
    defaults = {
        # Email configuration: Gmail normally uses smtp.gmail.com, port 587,
        # and a Google App Password (never a regular Gmail password).
        "teacher_email": "",
        "teacher_emails": [],
        "sender_email": "",
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "confidence_threshold": RECOGNITION_CONFIDENCE_THRESHOLD,
        "capture_sample_count": CAPTURE_SAMPLE_COUNT,
        "recognition_cooldown_seconds": RECOGNITION_COOLDOWN_SECONDS,
    }
    ensure_runtime_directories()
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        return defaults
    try:
        settings = {**defaults, **json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))}
        # Migrate the original single-recipient setting to the new list.
        if not settings.get("teacher_emails") and settings.get("teacher_email"):
            settings["teacher_emails"] = [settings["teacher_email"]]
        return settings
    except (OSError, json.JSONDecodeError):
        return defaults


def save_settings(settings: dict) -> None:
    """Persist non-secret user preferences."""
    # App passwords are intentionally never written to disk.
    safe_settings = {
        key: value for key, value in settings.items() if key != "smtp_password"
    }
    ensure_runtime_directories()
    SETTINGS_PATH.write_text(json.dumps(safe_settings, indent=2), encoding="utf-8")


def recognition_settings() -> dict:
    """Return validated, user-editable recognition tuning values."""
    settings = get_settings()
    return {
        "threshold": float(
            settings.get("confidence_threshold", RECOGNITION_CONFIDENCE_THRESHOLD)
        ),
        "samples": max(
            10, int(settings.get("capture_sample_count", CAPTURE_SAMPLE_COUNT))
        ),
        "cooldown": max(
            1,
            int(
                settings.get(
                    "recognition_cooldown_seconds", RECOGNITION_COOLDOWN_SECONDS
                )
            ),
        ),
    }
