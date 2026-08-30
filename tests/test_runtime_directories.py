"""Regression tests for runtime directory initialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config


class RuntimeDirectoriesTests(unittest.TestCase):
    def test_ensure_runtime_directories_creates_logs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset_dir = base / "dataset"
            models_dir = base / "models"
            database_dir = base / "database"
            exports_dir = base / "exports"
            logs_dir = base / "logs"

            with patch("config.BASE_DIR", base), \
                 patch("config.DATASET_DIR", dataset_dir), \
                 patch("config.MODELS_DIR", models_dir), \
                 patch("config.DATABASE_DIR", database_dir), \
                 patch("config.EXPORTS_DIR", exports_dir), \
                 patch("config.LOGS_DIR", logs_dir):
                config.ensure_runtime_directories()

            self.assertTrue(dataset_dir.exists())
            self.assertTrue(models_dir.exists())
            self.assertTrue(database_dir.exists())
            self.assertTrue(exports_dir.exists())
            self.assertTrue(logs_dir.exists())


if __name__ == "__main__":
    unittest.main()
