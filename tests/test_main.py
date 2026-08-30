"""UI validation tests for student identifiers."""

from __future__ import annotations

import unittest

import main


class MainUiTests(unittest.TestCase):
    def test_valid_student_id_examples(self) -> None:
        self.assertTrue(main.is_valid_student_id("1CE-001"))
        self.assertTrue(main.is_valid_student_id("CE-001"))
        self.assertFalse(main.is_valid_student_id("student"))
        self.assertFalse(main.is_valid_student_id(""))


if __name__ == "__main__":
    unittest.main()
