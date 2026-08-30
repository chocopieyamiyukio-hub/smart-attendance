"""Command-line model training entry point."""

from __future__ import annotations

import database
from config import ensure_runtime_directories
from recognition import train_model


def main() -> None:
    ensure_runtime_directories()
    database.init_db()
    try:
        samples, students = train_model()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error
    print(f"Training complete: {samples} face samples from {students} student(s).")


if __name__ == "__main__":
    main()
