"""Reset all face-recognition and attendance data for the application."""

from __future__ import annotations

import shutil

import database
from config import (DATASET_DIR, EXPORTS_DIR, LABELS_PATH, LOGS_DIR,
                    MODEL_PATH, MODELS_DIR, ensure_runtime_directories)


def reset_all_data() -> None:
    """Delete all registered students, attendance records, model artifacts, and dataset images."""
    ensure_runtime_directories()

    database.close_all_connections()
    database.delete_all_students_and_attendance()

    for path in (MODEL_PATH, LABELS_PATH):
        if path.exists():
            path.unlink()

    for directory in (DATASET_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR):
        if directory.exists():
            for child in list(directory.iterdir()):
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    elif child.is_file():
                        child.unlink()
                except PermissionError:
                    continue

    database.init_db()


def delete_student_data(student_id: str) -> bool:
    """Remove one student's database, face-image, and stale-model data."""
    database.close_all_connections()
    deleted = database.delete_student(student_id)
    student_folder = DATASET_DIR / student_id
    if student_folder.exists():
        shutil.rmtree(student_folder)
    # Labels refer to dataset folder names, so the saved model must never be
    # used after an ID change or deletion until retraining completes.
    for path in (MODEL_PATH, LABELS_PATH):
        if path.exists():
            path.unlink()
    return deleted


def move_student_dataset(old_student_id: str, new_student_id: str) -> None:
    """Keep face samples aligned with an edited student ID and invalidate LBPH."""
    if old_student_id != new_student_id:
        old_folder, new_folder = DATASET_DIR / old_student_id, DATASET_DIR / new_student_id
        if old_folder.exists():
            if new_folder.exists():
                raise ValueError("A dataset folder already exists for that Student ID.")
            old_folder.rename(new_folder)
    for path in (MODEL_PATH, LABELS_PATH):
        if path.exists():
            path.unlink()
