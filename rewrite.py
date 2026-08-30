import re

with open('recognition.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace _detector
code = re.sub(
    r'@lru_cache\(maxsize=1\)\ndef _detector\(\) -> cv2\.CascadeClassifier:.*?return detector',
    '''@lru_cache(maxsize=1)
def _detector():
    if not YUNET_PATH.exists():
        raise FileNotFoundError("YuNet model not found.")
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.9, 0.3, 5000)''',
    code,
    flags=re.DOTALL
)

# Replace detect_faces
code = re.sub(
    r'def detect_faces\([\s\S]*?\]',
    '''def detect_faces(
    frame: np.ndarray, detector=None
) -> list[tuple[int, int, int, int]]:
    detector = detector or _detector()
    height, width = frame.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    rects = []
    for face in faces:
        rects.append(tuple(map(int, face[:4])))
    return rects''',
    code,
    flags=re.DOTALL
)

# Replace _normalize_face_image
code = re.sub(
    r'def _normalize_face_image\(face: np\.ndarray\) -> np\.ndarray:[\s\S]*?return face',
    '''def _normalize_face_image(face: np.ndarray) -> np.ndarray:
    return face''',
    code,
    flags=re.DOTALL
)

# Replace prepare_face
code = re.sub(
    r'def prepare_face\(frame: np\.ndarray, rectangle: tuple\[int, int, int, int\]\) -> np\.ndarray:[\s\S]*?return face',
    '''def prepare_face(frame: np.ndarray, rectangle: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = rectangle
    height_limit, width_limit = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    width, height = min(width, width_limit - x), min(height, height_limit - y)
    if width < 40 or height < 40:
        raise ValueError("Detected face is too small.")
    crop = frame[y : y + height, x : x + width]
    face = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    return face''',
    code,
    flags=re.DOTALL
)

# Replace _load_recognizer with _load_sface
code = re.sub(
    r'def _load_recognizer\(\) -> cv2\.face_LBPHFaceRecognizer:[\s\S]*?return recognizer',
    '''@lru_cache(maxsize=1)
def _load_sface():
    if not SFACE_PATH.exists():
        raise FileNotFoundError("SFace model not found.")
    return cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")''',
    code,
    flags=re.DOTALL
)

# Remove _recognizer
code = re.sub(
    r'def _recognizer\(\) -> cv2\.face_LBPHFaceRecognizer:[\s\S]*?\n\n',
    '\n',
    code,
    flags=re.DOTALL
)

# Replace predict_face
code = re.sub(
    r'def predict_face\([\s\S]*?return student_id, confidence',
    '''def predict_face(
    face: np.ndarray,
    recognizer,
    label_mapping: dict[int, str],
    threshold: float,
) -> tuple[str | None, float]:
    # Extract feature
    feature = recognizer.feature(face)
    
    # Load embeddings
    if not MODEL_PATH.exists():
        return None, 1.0
    data = np.load(str(MODEL_PATH))
    labels = data['labels']
    embeddings = data['embeddings']
    
    best_student = None
    best_distance = 1.0
    
    # Compare against all known embeddings
    for i, emb in enumerate(embeddings):
        # Cosine distance
        dist = recognizer.match(feature, np.array([emb]), cv2.FaceRecognizerSF_FR_COSINE)
        if dist < best_distance:
            best_distance = dist
            best_student = label_mapping.get(labels[i])
            
    if best_student is None or best_distance > threshold:
        return None, best_distance
    return best_student, best_distance''',
    code,
    flags=re.DOTALL
)

# Replace train_model
code = re.sub(
    r'def train_model\(\) -> tuple\[int, int\]:[\s\S]*?return samples, students',
    '''def train_model() -> tuple[int, int]:
    ensure_runtime_directories()
    if not DATASET_DIR.exists():
        raise FileNotFoundError("Dataset directory is missing.")
    
    sface = _load_sface()
    faces = []
    labels = []
    label_mapping = {}
    current_label = 0
    
    student_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    for student_dir in student_dirs:
        student_id = student_dir.name
        label_mapping[current_label] = student_id
        
        student_embeddings = []
        for path in student_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                img = cv2.imread(str(path))
                if img is None: continue
                # We assume the dataset images are already cropped to FACE_SIZE
                feature = sface.feature(img)
                student_embeddings.append(feature[0])
                
        if student_embeddings:
            # Average the embeddings for the student to create a robust profile
            avg_emb = np.mean(student_embeddings, axis=0)
            faces.append(avg_emb)
            labels.append(current_label)
            current_label += 1

    if not faces:
        raise ValueError("No valid training images found.")

    LABELS_PATH.write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")
    
    np.savez_compressed(
        str(MODEL_PATH),
        embeddings=np.array(faces),
        labels=np.array(labels)
    )
    
    return sum(len(list(d.iterdir())) for d in student_dirs), len(student_dirs)''',
    code,
    flags=re.DOTALL
)

# Fix recognize_faces_live
code = code.replace(
    'recognizer = _load_recognizer()',
    'recognizer = _load_sface()'
)

with open('recognition.py', 'w', encoding='utf-8') as f:
    f.write(code)
