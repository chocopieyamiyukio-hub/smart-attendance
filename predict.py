"""Command-line live-attendance entry point."""

import database
from recognition import recognize_faces_live

if __name__ == "__main__":
    database.init_db()
    recognize_faces_live(database.mark_attendance)
