# Smart Attendance System

A local desktop application that registers student face samples, trains an OpenCV LBPH face-recognition model, records one attendance event per student per day in SQLite, and exports attendance to CSV. No cloud service, API key, or internet connection is needed after installation.

## Features

- Tkinter administrator dashboard
- Webcam face-sample capture with input validation
- OpenCV Haar detection and LBPH training/recognition
- SQLite student and attendance storage
- Duplicate attendance prevention at the database level
- CSV report export
- Command-line training and recognition entry points

## Project layout

```text
smart_attendance_system/
├── app.py              # GUI entry point
├── main.py             # GUI implementation
├── train.py            # CLI training entry point
├── predict.py          # CLI live-recognition entry point
├── recognition.py      # Detection, training, and prediction pipeline
├── database.py         # SQLite and CSV operations
├── config.py           # Paths and runtime settings
├── requirements.txt
├── tests/test_database.py
├── dataset/            # Created at runtime: <student_id>/*.jpg
├── models/             # Created at runtime: trained model and labels
├── database/           # Created at runtime: SQLite database
└── exports/            # Created at runtime: CSV reports
```

## Requirements

- Python 3.10 or later (tested with Python 3.12)
- A functioning webcam available to the local user
- Windows, macOS, or Linux desktop session with GUI access

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## How to use

1. Enter a student ID (letters, numbers, `_`, `-`) and name.
2. Choose **Capture Face Data** and keep one face clearly visible until 50 samples are collected. Press `Q` to cancel without changing existing data.
3. Choose **Train Recognition Model** after registering one or more students.
4. Choose **Start Live Attendance**. Recognized students are stored only once for the current calendar day. Press `Q` in the camera window to stop.
5. Choose **Export Logs (CSV)** to create `exports/attendance_report.csv`.

Training can also be run from the terminal after captures exist:

```bash
python train.py
```

To launch just live recognition with an existing model:

```bash
python predict.py
```

## Data and model behavior

The application creates its own dataset as face crops under `dataset/<student_id>/`; no bundled dataset is needed. Training rejects unreadable images and images with no detectable face, and fails with a clear message if no usable samples remain. The trained LBPH model and its ID-label mapping are saved under `models/` and must be retrained when samples change.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

- **`cv2.face` unavailable:** remove `opencv-python` and install the provided requirements; LBPH is supplied by `opencv-contrib-python`.
- **Camera cannot be opened:** close other applications using it and grant desktop-camera permission to Python.
- **No face detected during capture:** improve lighting, look directly at the camera, and keep your face large and unobstructed.
- **Unknown recognition:** capture varied, clear samples and retrain. The application is intended for attendance assistance; verify records for decisions with real-world consequences.
