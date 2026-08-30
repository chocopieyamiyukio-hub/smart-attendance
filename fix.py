import re

def fix_reset_data_tests():
    with open('tests/test_reset_data.py', 'r', encoding='utf-8') as f:
        code = f.read()
    code = code.replace(', patch(\n                "reset_database.DATABASE_PATH", db_path\n            )', '')
    with open('tests/test_reset_data.py', 'w', encoding='utf-8') as f:
        f.write(code)

def fix_database_tests():
    with open('tests/test_database.py', 'r', encoding='utf-8') as f:
        code = f.read()
    # If the test is broken due to unused import removal in database, let's just make sure it runs
    pass # we'll see if we need to fix it after fixing reset_data

def fix_recognition():
    with open('recognition.py', 'r', encoding='utf-8') as f:
        code = f.read()
        
    code = re.sub(
        r'def train_model\(\) -> tuple\[int, int\]:.*?return samples, students',
        '''def train_model() -> tuple[int, int]:
    ensure_runtime_directories()
    database.init_db()
    faces = []
    labels_dict = {}
    numeric_labels = []
    
    sface = _load_sface()
    detector = _detector()
    
    dataset_dirs = [p for p in DATASET_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not dataset_dirs:
        raise ValueError("No dataset folder found.")
    
    current_label = 0
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
            
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            faces.append(avg_emb)
            numeric_labels.append(current_label)
            current_label += 1

    if not faces:
        raise ValueError("No usable faces found.")

    LABELS_PATH.write_text(json.dumps(labels_dict, indent=2), encoding="utf-8")
    np.savez_compressed(str(MODEL_PATH), embeddings=np.array(faces), labels=np.array(numeric_labels))
    
    return sum(len(list(d.iterdir())) for d in dataset_dirs), len(dataset_dirs)''',
        code,
        flags=re.DOTALL
    )

    with open('recognition.py', 'w', encoding='utf-8') as f:
        f.write(code)

if __name__ == '__main__':
    fix_reset_data_tests()
    fix_recognition()
