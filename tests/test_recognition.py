"""End-to-end model-artifact test using a synthetic face sample."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import recognition


class WholeImageDetector:
    """Test-only detector that treats the synthetic image as one face."""

    def detectMultiScale(self, image, *args, **kwargs):  # noqa: N802 - OpenCV API spelling
        height, width = image.shape[:2]
        return np.array([[0, 0, width, height]])


class RecognitionTests(unittest.TestCase):
    def test_training_writes_loadable_model_and_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            student = dataset / "S-001"
            student.mkdir(parents=True)
            image = np.full((200, 200, 3), 125, dtype=np.uint8)
            cv2.circle(image, (100, 100), 60, (180, 180, 180), -1)
            self.assertTrue(cv2.imwrite(str(student / "sample.jpg"), image))
            model = root / "model.yml"
            labels = root / "labels.json"
            with patch("recognition.DATASET_DIR", dataset), patch("recognition.MODEL_PATH", model), patch("recognition.LABELS_PATH", labels), patch("recognition._detector", return_value=WholeImageDetector()):
                samples, students = recognition.train_model()
                self.assertEqual((samples, students), (1, 1))
                self.assertTrue(model.is_file())
                self.assertEqual(json.loads(labels.read_text(encoding="utf-8")), {"S-001": 0})
                loaded = recognition._recognizer()
                loaded.read(str(model))

    def test_add_face_samples_for_student_saves_unique_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir(parents=True)
            valid_image = np.full((220, 220, 3), 120, dtype=np.uint8)
            cv2.circle(valid_image, (110, 110), 70, (190, 190, 190), -1)
            first_path = Path(directory) / "first.jpg"
            second_path = Path(directory) / "second.jpg"
            self.assertTrue(cv2.imwrite(str(first_path), valid_image))
            self.assertTrue(cv2.imwrite(str(second_path), valid_image))
            with patch("recognition.DATASET_DIR", dataset), patch("recognition._detector", return_value=WholeImageDetector()):
                stats = recognition.add_face_samples_for_student("S-001", [first_path, second_path])
            self.assertEqual(stats["images_selected"], 2)
            self.assertEqual(stats["valid_faces"], 2)
            self.assertEqual(stats["new_images_added"], 2)
            student_folder = dataset / "S-001"
            saved_files = sorted(path.name for path in student_folder.iterdir())
            self.assertEqual(saved_files, ["new001.jpg", "new002.jpg"])

    def test_largest_face_selection_prefers_the_biggest_region(self) -> None:
        rectangles = [(0, 0, 120, 120), (20, 20, 40, 40), (5, 5, 90, 90)]
        self.assertEqual(recognition._largest_face_rectangles(rectangles), (0, 0, 120, 120))


if __name__ == "__main__":
    unittest.main()
