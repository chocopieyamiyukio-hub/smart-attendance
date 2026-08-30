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
                    FACE_SIZE, IMAGE_EXTENSIONS, LABELS_PATH, 
                    MIN_FACE_SHARPNESS, MODEL_PATH, RECOGNITION_LOG_PATH, YUNET_PATH, SFACE_PATH,
                    ensure_runtime_directories, recognition_settings)

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _detector():
    if not YUNET_PATH.exists():
        raise FileNotFoundError("YuNet model not found.")
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.9, 0.3, 5000)


def detect_faces(
    frame: np.ndarray, detector=None
) -> list[tuple[int, int, int, int]]:
    """Detect credible faces using YuNet."""
    detector = detector or _detector()
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    rects = []
    for face in faces:
        rects.append(tuple(map(int, face[:4])))
    return rects


def face_sharpness(face: np.ndarray) -> float:
    """Variance of the Laplacian; low values indicate blurred images."""
    return float(cv2.Laplacian(face, cv2.CV_64F).var())


def _largest_face_rectangles(
    rectangles: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Return the largest detected face rectangle based on area."""
    if not rectangles:
        raise ValueError("No detected faces were found.")
    return max(rectangles, key=lambda rectangle: rectangle[2] * rectangle[3])


def _normalize_face_image(face: np.ndarray) -> np.ndarray:
    return face


def _load_label_mapping() -> dict[int, str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError("Label mapping file is missing. Train the model first.")
    mapping = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Label mapping is malformed.")
    return {int(k): str(v) for k, v in mapping.items()}


@lru_cache(maxsize=1)
def _load_sface():
    if not SFACE_PATH.exists():
        raise FileNotFoundError("SFace model not found.")
    return cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")


def _face_quality_ok(face: np.ndarray) -> bool:
    return face_sharpness(face) >= MIN_FACE_SHARPNESS


def predict_face(
    face: np.ndarray,
    recognizer,
    label_mapping: dict[int, str],
    threshold: float,
) -> tuple[str | None, float]:
    feature = recognizer.feature(face)
    
    if not MODEL_PATH.exists():
        return None, 0.0
    with np.load(str(MODEL_PATH)) as data:
        labels = data['labels']
        embeddings = data['embeddings']
    
    best_student = None
    best_score = 0.0
    
    for i, emb in enumerate(embeddings):
        score = recognizer.match(feature, np.array([emb]), cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = score
            best_student = label_mapping.get(labels[i])
            
    if best_student is None or best_score < threshold:
        return None, best_score
    return best_student, best_score


def prepare_face(frame: np.ndarray, rectangle: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rectangle
    height_limit, width_limit = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    width, height = min(width, width_limit - x), min(height, height_limit - y)
    if width < 40 or height < 40:
        raise ValueError("Detected face is too small.")
    crop = frame[y : y + height, x : x + width]
    face = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    return face




def log_recognition(student_id: str, name: str, confidence: float, result: str) -> None:
    ensure_runtime_directories()
    new_file = not RECOGNITION_LOG_PATH.exists()
    with RECOGNITION_LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(
                ("time", "student_id", "student_name", "confidence", "result")
            )
        writer.writerow(
            (
                datetime.now().isoformat(timespec="seconds"),
                student_id,
                name,
                f"{confidence:.2f}",
                result,
            )
        )


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


def add_face_samples_for_student(
    student_id: str,
    image_paths: list[Path | str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Add new face samples for an existing student without changing their identity."""
    ensure_runtime_directories()
    if not student_id:
        raise ValueError("A student ID is required.")
    try:
        database.init_db()
        if not database.student_exists(student_id):
            LOGGER.warning(
                "Student %s was not found in the database; continuing with dataset update.",
                student_id,
            )
    except (
        Exception
    ) as error:  # pragma: no cover - defensive guard for offline/test environments
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
        # Read with PIL to handle mobile EXIF rotation correctly
        try:
            from PIL import Image, ImageOps
            import numpy as np
            pil_img = Image.open(str(image_path)).convert("RGB")
            pil_img = ImageOps.exif_transpose(pil_img)
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            image = cv2.imread(str(image_path))
            
        if image is None:
            stats["ignored_images"] += 1
            continue
            
        # Fast downscale for very large images to drastically speed up YuNet detection
        height, width = image.shape[:2]
        max_dim = 1280
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            small_image = cv2.resize(image, (int(width * scale), int(height * scale)))
            
            # Temporarily lower threshold for downscaled detection
            original_threshold = detector.getScoreThreshold()
            detector.setScoreThreshold(0.6)
            small_rects = detect_faces(small_image, detector)
            detector.setScoreThreshold(original_threshold)
            
            rectangles = []
            for (x, y, w, h) in small_rects:
                rectangles.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
        else:
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
    for student_dir in sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ):
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
    ensure_runtime_directories()
    database.init_db()
    
    sface = _load_sface()
    detector = _detector()
    
    dataset_dirs = [p for p in DATASET_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not dataset_dirs:
        raise ValueError("No dataset folder found.")
    
    faces = []
    numeric_labels = []
    labels_dict = {}
    current_label = 0
    sample_count = 0
    
    for student_dir in dataset_dirs:
        student_id = student_dir.name
        labels_dict[current_label] = student_id
        
        embeddings = []
        for img_path in sorted(student_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS: continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            if img.shape[:2] == (FACE_SIZE[1], FACE_SIZE[0]):
                face = img
            else:
                rects = detect_faces(img, detector)
                if not rects: continue
                try:
                    face = prepare_face(img, _largest_face_rectangles(rects))
                except ValueError:
                    continue
            
            emb = sface.feature(face)
            embeddings.append(emb[0])
            sample_count += 1
            
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            faces.append(avg_emb)
            numeric_labels.append(current_label)
            current_label += 1

    if not faces:
        raise ValueError("No usable faces found.")

    LABELS_PATH.write_text(json.dumps(labels_dict, indent=2), encoding="utf-8")
    np.savez_compressed(str(MODEL_PATH), embeddings=np.array(faces), labels=np.array(numeric_labels))
    
    return sample_count, len(dataset_dirs)

def model_available() -> bool:
    return MODEL_PATH.is_file() and LABELS_PATH.is_file()


def recognize_faces_live(mark_attendance_callback: Callable[[str], bool]) -> None:
    """Backward-compatible OpenCV live loop with quality checks and cooldown."""
    if not model_available():
        raise FileNotFoundError(
            "No trained model found. Train the recognition model first."
        )
    recognizer = _load_sface()
    label_mapping = _load_label_mapping()
    detector = _detector()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                    LOGGER.debug(
                        "Face candidate rejected by preprocessing: %s",
                        (x, y, width, height),
                    )
                    continue
                student_id, confidence = predict_face(
                    face, recognizer, label_mapping, settings["threshold"]
                )
                allowed = bool(
                    student_id
                    and time.monotonic() - last_seen.get(student_id, 0)
                    >= settings["cooldown"]
                )
                if allowed:
                    mark_attendance_callback(student_id)
                    last_seen[student_id] = time.monotonic()
                text = student_id or "Unknown Face"
                color = (0, 180, 0) if student_id else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    frame,
                    f"{text} ({confidence:.1f})",
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
                log_recognition(
                    student_id or "",
                    student_id or "Unknown",
                    confidence,
                    "Recognized" if student_id else "Unknown",
                )
            cv2.imshow("Smart Attendance System (press Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
