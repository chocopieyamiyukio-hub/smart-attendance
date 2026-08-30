"""Face preprocessing, LBPH training, and reliable live recognition."""

from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import database
from config import (CAMERA_INDEX, DATASET_DIR, FACE_DETECTION_MIN_NEIGHBORS,
                    FACE_DETECTION_MIN_SIZE, FACE_DETECTION_SCALE_FACTOR,
                    FACE_SIZE, IMAGE_EXTENSIONS, LABELS_PATH, LBPH_GRID_X,
                    LBPH_GRID_Y, LBPH_NEIGHBORS, LBPH_RADIUS,
                    MIN_FACE_SHARPNESS, MODEL_PATH, RECOGNITION_LOG_PATH,
                    ensure_runtime_directories, recognition_settings)

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _detector() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled detector using conservative false-positive settings.

    Haar remains the compatible fallback for offline installations. The
    preprocessing below (equalisation, size checks and blur rejection) makes
    it materially more reliable while keeping existing OpenCV deployments.
    """
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError("OpenCV's Haar cascade classifier could not be loaded.")
    return detector


def detect_faces(frame: np.ndarray, detector: cv2.CascadeClassifier | None = None) -> list[tuple[int, int, int, int]]:
    """Detect credible faces after lighting normalization."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    classifier = detector or _detector()
    return [tuple(map(int, rect)) for rect in classifier.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION_SCALE_FACTOR,
        minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
        minSize=FACE_DETECTION_MIN_SIZE,
    )]


def face_sharpness(face: np.ndarray) -> float:
    """Variance of the Laplacian; low values indicate blurred images."""
    return float(cv2.Laplacian(face, cv2.CV_64F).var())


def _largest_face_rectangles(rectangles: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Return the largest detected face rectangle based on area."""
    if not rectangles:
        raise ValueError("No detected faces were found.")
    return max(rectangles, key=lambda rectangle: rectangle[2] * rectangle[3])


def _normalize_face_image(face: np.ndarray) -> np.ndarray:
    """Normalize brightness and contrast to make uploaded photos more consistent."""
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    return face


def _load_label_mapping() -> dict[int, str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError("Label mapping file is missing. Train the model first.")
    mapping = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Label mapping is malformed.")
    return {int(value): key for key, value in mapping.items()}


def _load_recognizer() -> cv2.face_LBPHFaceRecognizer:
    recognizer = _recognizer()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Recognition model not found. Train the model first.")
    recognizer.read(str(MODEL_PATH))
    return recognizer


def _face_quality_ok(face: np.ndarray) -> bool:
    return face_sharpness(face) >= MIN_FACE_SHARPNESS


def predict_face(face: np.ndarray, recognizer: cv2.face_LBPHFaceRecognizer,
                 label_mapping: dict[int, str], threshold: float) -> tuple[str | None, float]:
    label, confidence = recognizer.predict(face)
    student_id = label_mapping.get(label)
    if student_id is None or confidence > threshold:
        return None, confidence
    return student_id, confidence


def prepare_face(frame: np.ndarray, rectangle: tuple[int, int, int, int]) -> np.ndarray:
    """Crop, equalize and resize one quality-checked face for LBPH."""
    x, y, width, height = rectangle
    height_limit, width_limit = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    width, height = min(width, width_limit - x), min(height, height_limit - y)
    if width < 40 or height < 40:
        raise ValueError("Detected face is too small.")
    crop = frame[y:y + height, x:x + width]
    face = _normalize_face_image(crop)
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    face = cv2.equalizeHist(face)
    if not _face_quality_ok(face):
        raise ValueError("Face image is too blurry; please hold still and improve lighting.")
    return face


def _recognizer() -> cv2.face_LBPHFaceRecognizer:
    """Return tuned LBPH recognizer. Tune constants in config.py if required."""
    if not hasattr(cv2, "face"):
        raise RuntimeError("OpenCV face module is unavailable. Install opencv-contrib-python.")
    return cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS, neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X, grid_y=LBPH_GRID_Y,
    )

def log_recognition(student_id: str, name: str, confidence: float, result: str) -> None:
    ensure_runtime_directories()
    new_file = not RECOGNITION_LOG_PATH.exists()
    with RECOGNITION_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(("time", "student_id", "student_name", "confidence", "result"))
        writer.writerow((datetime.now().isoformat(timespec="seconds"), student_id, name, f"{confidence:.2f}", result))


def _unique_dataset_path(student_dir: Path, source_path: Path) -> Path:
    """Return a non-conflicting filename inside the student's dataset folder."""
    suffix = source_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {source_path.suffix}")
    counter = 1
    while True:
        candidate = student_dir / f"new{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def add_face_samples_for_student(student_id: str, image_paths: list[Path | str], progress_callback: Callable[[int, int], None] | None = None) -> dict[str, int]:
    """Add new face samples for an existing student without changing their identity."""
    ensure_runtime_directories()
    if not student_id:
        raise ValueError("A student ID is required.")
    try:
        database.init_db()
        if not database.student_exists(student_id):
            LOGGER.warning("Student %s was not found in the database; continuing with dataset update.", student_id)
    except Exception as error:  # pragma: no cover - defensive guard for offline/test environments
        LOGGER.warning("Could not validate student record %s: %s", student_id, error)

    student_dir = DATASET_DIR / student_id
    student_dir.mkdir(parents=True, exist_ok=True)
    detector = _detector()
    stats = {
        "images_selected": len(image_paths),
        "valid_faces": 0,
        "ignored_images": 0,
        "new_images_added": 0,
    }

    for index, raw_path in enumerate(image_paths, start=1):
        if progress_callback is not None:
            progress_callback(index, len(image_paths))
        image_path = Path(raw_path)
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            stats["ignored_images"] += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            stats["ignored_images"] += 1
            continue
        rectangles = detect_faces(image, detector)
        if not rectangles:
            stats["ignored_images"] += 1
            continue
        try:
            face = prepare_face(image, _largest_face_rectangles(rectangles))
        except ValueError:
            stats["ignored_images"] += 1
            continue
        output_path = _unique_dataset_path(student_dir, image_path)
        saved = cv2.imwrite(str(output_path), face)
        if not saved:
            stats["ignored_images"] += 1
            continue
        stats["valid_faces"] += 1
        stats["new_images_added"] += 1

    return stats


def _validate_dataset() -> tuple[int, int, list[str]]:
    if not DATASET_DIR.exists():
        return 0, 0, []
    missing_students: list[str] = []
    sample_count = 0
    student_count = 0
    for student_dir in sorted(path for path in DATASET_DIR.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if not student_dir.name:
            continue
        if not database.student_exists(student_dir.name):
            missing_students.append(student_dir.name)
        student_count += 1
        for image_path in student_dir.iterdir():
            if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                sample_count += 1
    return sample_count, student_count, missing_students


def train_model() -> tuple[int, int]:
    """Train from quality-controlled dataset images and return sample/student counts."""
    ensure_runtime_directories()
    database.init_db()
    faces: list[np.ndarray] = []
    numeric_labels: list[int] = []
    labels: dict[str, int] = {}
    detector = _detector()
    dataset_dirs = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not dataset_dirs:
        raise ValueError("No dataset folder found. Add face images before training.")
    sample_count, student_count, missing_students = _validate_dataset()
    if missing_students:
        LOGGER.warning(
            "Datasets found for unregistered students: %s. These folders will be skipped unless the student is registered.",
            missing_students,
        )
    LOGGER.info("Dataset validation found %s sample(s) in %s student folder(s).", sample_count, student_count)
    for student_dir in dataset_dirs:
        if not database.student_exists(student_dir.name):
            LOGGER.warning(
                "Dataset folder %s has no matching registered student. Including it in training, but register the student to enable attendance recording.",
                student_dir.name,
            )
        accepted = 0
        for image_path in sorted(student_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                LOGGER.debug("Skipping unsupported file %s", image_path)
                continue
            if not image_path.is_file():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                LOGGER.warning("Skipping unreadable image: %s", image_path)
                continue
            rectangles = detect_faces(image, detector)
            if not rectangles:
                LOGGER.warning("Skipping image without a detectable face: %s", image_path)
                continue
            try:
                face = prepare_face(image, _largest_face_rectangles(rectangles))
            except ValueError as error:
                LOGGER.warning("Skipping invalid face image %s: %s", image_path, error)
                continue
            if student_dir.name not in labels:
                labels[student_dir.name] = len(labels)
            faces.append(face)
            numeric_labels.append(labels[student_dir.name])
            accepted += 1
        LOGGER.info("Accepted %s samples for %s", accepted, student_dir.name)
    if not faces:
        raise ValueError("No usable faces found. Capture clear, well-lit face images first.")
    recognizer = _recognizer()
    recognizer.train(faces, np.asarray(numeric_labels, dtype=np.int32))
    recognizer.save(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return len(faces), len(labels)


def model_available() -> bool:
    return MODEL_PATH.is_file() and LABELS_PATH.is_file()


def recognize_faces_live(mark_attendance_callback: Callable[[str], bool]) -> None:
    """Backward-compatible OpenCV live loop with quality checks and cooldown."""
    if not model_available():
        raise FileNotFoundError("No trained model found. Train the recognition model first.")
    recognizer = _load_recognizer()
    label_mapping = _load_label_mapping()
    detector = _detector()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_seen: dict[str, float] = {}
    settings = recognition_settings()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The webcam did not return a frame.")
            for x, y, width, height in detect_faces(frame, detector):
                try:
                    face = prepare_face(frame, (x, y, width, height))
                except ValueError:
                    LOGGER.debug("Face candidate rejected by preprocessing: %s", (x, y, width, height))
                    continue
                student_id, confidence = predict_face(face, recognizer, label_mapping, settings["threshold"])
                allowed = bool(student_id and time.monotonic() - last_seen.get(student_id, 0) >= settings["cooldown"])
                if allowed:
                    mark_attendance_callback(student_id)
                    last_seen[student_id] = time.monotonic()
                text = student_id or "Unknown Face"
                color = (0, 180, 0) if student_id else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, f"{text} ({confidence:.1f})", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
                log_recognition(student_id or "", student_id or "Unknown", confidence, "Recognized" if student_id else "Unknown")
            cv2.imshow("Smart Attendance System (press Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release(); cv2.destroyAllWindows()
