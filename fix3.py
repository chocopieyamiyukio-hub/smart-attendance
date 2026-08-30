with open('tests/test_reset_data.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('tests/test_reset_data.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if 'reset_database.DATABASE_PATH' not in line:
            f.write(line)

with open('tests/test_recognition.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace('data = np.load(str(model))\n                self.assertIn("embeddings", data)', 'with np.load(str(model)) as data:\n                    self.assertIn("embeddings", data)')
with open('tests/test_recognition.py', 'w', encoding='utf-8') as f:
    f.write(code)

with open('recognition.py', 'r', encoding='utf-8') as f:
    code = f.read()

# We need to make sure predict_face closes the npz file too!
code = code.replace("data = np.load(str(MODEL_PATH))", "with np.load(str(MODEL_PATH)) as data:\n        labels = data['labels']\n        embeddings = data['embeddings']")
code = code.replace("labels = data['labels']\n    embeddings = data['embeddings']", "")

with open('recognition.py', 'w', encoding='utf-8') as f:
    f.write(code)
