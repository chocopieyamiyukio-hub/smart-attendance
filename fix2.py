with open('tests/test_reset_data.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(', patch(\n                "reset_database.DATABASE_PATH", db_path\n            )', '')

with open('tests/test_reset_data.py', 'w', encoding='utf-8') as f:
    f.write(code)

with open('recognition.py', 'r', encoding='utf-8') as f:
    code = f.read()

import re
import numpy as np
import cv2
import json

start = code.find('def train_model() -> tuple[int, int]:')
end = code.find('def model_available() -> bool:')
new_train = '''def train_model() -> tuple[int, int]:
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

'''
code = code[:start] + new_train + code[end:]

with open('recognition.py', 'w', encoding='utf-8') as f:
    f.write(code)
