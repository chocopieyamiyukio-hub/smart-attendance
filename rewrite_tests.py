import re

with open('tests/test_recognition.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace WholeImageDetector
code = re.sub(
    r'class WholeImageDetector:[\s\S]*?return np\.array\(\[\[0, 0, width, height\]\]\)',
    '''class WholeImageDetector:
    """Test-only detector that treats the synthetic image as one face."""
    
    def setInputSize(self, size):
        pass

    def detect(self, image):
        height, width = image.shape[:2]
        # YuNet returns (retval, array of [x, y, w, h, ...])
        return True, np.array([[0, 0, width, height, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])''',
    code,
    flags=re.DOTALL
)

code = code.replace('model = root / "model.yml"', 'model = root / "embeddings.npz"')
code = code.replace('{"S-001": 0}', '{"0": "S-001"}')
code = code.replace('loaded = recognition._recognizer()\n                loaded.read(str(model))', 'data = np.load(str(model))\n                self.assertIn("embeddings", data)')

with open('tests/test_recognition.py', 'w', encoding='utf-8') as f:
    f.write(code)
