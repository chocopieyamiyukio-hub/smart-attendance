"""Tests for resetting the registered face-recognition data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import database
import reset_database


class ResetDataTests(unittest.TestCase):
    def test_reset_all_data_removes_artifacts_and_recreates_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset_dir = base / "dataset"
            models_dir = base / "models"
            database_dir = base / "database"
            db_path = database_dir / "attendance.db"
            model_path = models_dir / "face_recognizer.yml"
            labels_path = models_dir / "labels.json"

            dataset_dir.mkdir(parents=True, exist_ok=True)
            (dataset_dir / "student-001").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "student-001" / "sample.jpg").write_bytes(b"fake-image")
            models_dir.mkdir(parents=True, exist_ok=True)
            database_dir.mkdir(parents=True, exist_ok=True)
            db_path.write_bytes(b"old-db")
            model_path.write_text("old-model", encoding="utf-8")
            labels_path.write_text("{}", encoding="utf-8")

            with patch("reset_database.DATASET_DIR", dataset_dir), \
                 patch("reset_database.MODELS_DIR", models_dir), \
                 patch("reset_database.DATABASE_PATH", db_path), \
                 patch("reset_database.MODEL_PATH", model_path), \
                 patch("reset_database.LABELS_PATH", labels_path), \
                 patch("database.DATABASE_PATH", db_path):
                reset_database.reset_all_data()

            self.assertTrue(dataset_dir.exists())
            self.assertEqual(list(dataset_dir.iterdir()), [])
            self.assertFalse(model_path.exists())
            self.assertFalse(labels_path.exists())
            self.assertTrue(db_path.exists())
            self.assertTrue(database.student_exists("placeholder") is False)


if __name__ == "__main__":
    unittest.main()
