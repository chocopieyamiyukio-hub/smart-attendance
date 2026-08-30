"""Integration tests for the SQLite attendance layer."""

from __future__ import annotations

import gc
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import database


class DatabaseTests(unittest.TestCase):
    def test_student_attendance_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "attendance.db"
            report_path = Path(directory) / "report.csv"
            with patch("database.DATABASE_PATH", db_path), patch(
                "database.EXPORTS_DIR", Path(directory)
            ):
                database.init_db()
                database.add_student("S-001", "Ada Lovelace")
                self.assertTrue(
                    database.mark_attendance("S-001", datetime(2026, 8, 5, 9, 0))
                )
                self.assertFalse(
                    database.mark_attendance("S-001", datetime(2026, 8, 5, 9, 30))
                )
                self.assertTrue(
                    database.mark_attendance("S-001", datetime(2026, 8, 5, 13, 30))
                )
                self.assertTrue(
                    database.mark_attendance("S-001", datetime(2026, 8, 6, 9, 0))
                )
                database.export_to_csv(report_path, report_date="2026-08-05")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("8:45to10:45,S-001,Ada Lovelace", report)
            self.assertIn("13:00to14:00,S-001,Ada Lovelace", report)
            self.assertNotIn("2026-08-06", report)
            # Explicit collection avoids delayed SQLite handle cleanup on Windows.
            gc.collect()

    def test_update_and_delete_student_preserves_or_removes_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "attendance.db"
            with patch("database.DATABASE_PATH", db_path):
                database.init_db()
                database.add_student("S-001", "Ada")
                database.mark_attendance("S-001", datetime(2026, 8, 5, 9, 0))
                database.update_student("S-001", "S-002", "Ada Lovelace")
                self.assertEqual(
                    database.get_student("S-002"), ("S-002", "Ada Lovelace")
                )
                self.assertIsNone(database.get_student("S-001"))
                self.assertTrue(database.delete_student("S-002"))
                self.assertEqual(database.dashboard_summary()["records"], 0)


if __name__ == "__main__":
    unittest.main()
