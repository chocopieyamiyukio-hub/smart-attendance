"""Premium CustomTkinter interface for the Smart Attendance System."""

from __future__ import annotations

import json
import logging
import queue
import re
import smtplib
import socket
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image, ImageOps
from ttkbootstrap import Style

import database
import recognition
import reset_database
from config import (BASE_DIR, CAMERA_INDEX, DATASET_DIR, EXPORTS_DIR,
                    IMAGE_EXTENSIONS, LABELS_PATH, MODEL_PATH,
                    configure_logging, ensure_runtime_directories,
                    get_settings, recognition_settings, save_settings)
from services.email_service import (build_email, generate_attendance_csv,
                                    send_email)

LOGGER = logging.getLogger(__name__)
SAFE_ID = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+-[A-Za-z0-9]+$")
COLORS = {
    "navy": "#1E3A5F",
    "blue": "#2563EB",
    "green": "#10B981",
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "text": "#1F2937",
    "muted": "#64748B",
    "orange": "#F59E0B",
    "red": "#D13232",
    "purple": "#7A39EB",
    "soft_blue": "#EEF2FF",
    "line": "#E5EAF2",
}
PAGES = [
    ("🏠", "Home"),
    ("👤", "Registration"),
    ("📷", "Capture"),
    ("🎥", "Attendance"),
    ("📊", "Reports"),
]


def is_valid_student_id(student_id: str) -> bool:
    """Return True when the student ID matches the expected format like 1CE-001."""
    return bool(student_id) and bool(SAFE_ID.fullmatch(student_id.strip()))


class AttendanceApp(ctk.CTk):
    """Application shell; widgets delegate persistence and vision to existing modules."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        Style(theme="flatly")
        super().title("Face Recognition Smart Attendance")
        self.geometry("1360x940")
        self.minsize(1160, 780)
        self.configure(fg_color=COLORS["bg"])
        self.student_id = ""
        self.student_name = ""
        self.current_page = 0
        self.trained = recognition.model_available()
        self.training_required = bool(database.list_students()) and not self.trained
        self.camera_stop = threading.Event()
        self.capture_stop = threading.Event()
        self.camera_queue: queue.Queue = queue.Queue(maxsize=2)
        self.recognition_cooldowns: dict[str, float] = {}
        self.live_attendance_records: list[tuple[str, str, str, str]] = []
        self._build_shell()
        self.show_page(0)

    def _build_shell(self) -> None:
        # A wider navigation rail keeps labels comfortably readable in demos.
        self.sidebar = ctk.CTkFrame(
            self, width=300, corner_radius=0, fg_color=COLORS["navy"]
        )
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_propagate(False)
        ctk.CTkLabel(
            self.sidebar,
            text="◉ SMART ATTENDANCE",
            font=("Segoe UI", 13, "bold"),
            text_color="white",
        ).pack(pady=(30, 42))
        self.nav_buttons = []
        self.nav_page_indices = []
        for index, (icon, title) in enumerate(PAGES):
            button = ctk.CTkButton(
                self.sidebar,
                text=f" {icon}   {title}",
                anchor="w",
                height=42,
                font=("Segoe UI", 13),
                fg_color="transparent",
                hover_color="#315A86",
                command=lambda i=index: self.show_page(i),
            )
            button.pack(fill="x", padx=15, pady=4)
            self.nav_buttons.append(button)
            self.nav_page_indices.append(index)
        ctk.CTkLabel(
            self.sidebar,
            text="Fifth Year Project\nComputer Engineering",
            justify="left",
            text_color="#B8CAE0",
            font=("Segoe UI", 11),
        ).pack(side="bottom", padx=22, pady=26, anchor="w")
        self.header = ctk.CTkFrame(self, height=78, corner_radius=0, fg_color="white")
        self.header.grid(row=0, column=1, sticky="new")
        self.header.grid_propagate(False)
        self.page_title = ctk.CTkLabel(
            self.header,
            text="",
            font=("Segoe UI", 23, "bold"),
            text_color=COLORS["text"],
        )
        self.page_title.pack(side="left", padx=34, pady=18)
        self.step_label = ctk.CTkLabel(
            self.header, text="", font=("Segoe UI", 12), text_color=COLORS["muted"]
        )
        self.step_label.pack(side="right", padx=32)
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=1, sticky="nsew", padx=34, pady=25)
        self.status = ctk.CTkLabel(
            self,
            text="●  Ready",
            height=28,
            anchor="w",
            fg_color="#EAF2FB",
            text_color=COLORS["navy"],
            font=("Segoe UI", 11),
        )
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def notify(self, message: str, kind: str = "success") -> None:
        color = (
            COLORS["green"]
            if kind == "success"
            else COLORS["red"] if kind == "error" else COLORS["orange"]
        )
        self.status.configure(text=f"●  {message}", text_color=color)

    def show_page(self, page: int) -> None:
        self.stop_camera()
        self.capture_stop.set()
        self._close_capture_preview()
        self.current_page = page
        self._set_app_chrome(page != 0)
        for child in self.content.winfo_children():
            child.destroy()
        for page_index, button in zip(self.nav_page_indices, self.nav_buttons):
            button.configure(
                fg_color="#315A86" if page_index == page else "transparent"
            )
        self.page_title.configure(text=PAGES[page][1])
        self.step_label.configure(text=f"STEP {page + 1} OF 5")
        builders = [
            self.welcome_page,
            self.registration_page,
            self.capture_page,
            self.attendance_page,
            self.reports_page,
        ]
        builders[page]()

    def _set_app_chrome(self, visible: bool) -> None:
        """Hide navigation on the hero welcome screen and restore it afterward."""
        binding = getattr(self, "_welcome_resize_binding", None)
        if binding and self.content.winfo_exists():
            self.content.unbind("<Configure>", binding)
            self._welcome_resize_binding = None
        self._welcome_background_label = None
        self._welcome_background_ctk = None
        if visible:
            self.sidebar.grid()
            self.header.grid()
            self.content.grid_configure(column=1, columnspan=1)
        else:
            self.sidebar.grid_remove()
            self.header.grid_remove()
            self.content.grid_configure(column=0, columnspan=2)

    def card(self, parent, **kwargs):
        return ctk.CTkFrame(
            parent,
            corner_radius=18,
            fg_color="white",
            border_width=1,
            border_color=COLORS["line"],
            **kwargs,
        )

    def page_heading(self, parent, heading: str, text: str = "") -> None:
        ctk.CTkLabel(
            parent,
            text=heading,
            font=("Segoe UI", 28, "bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", pady=(0, 5))
        if text:
            ctk.CTkLabel(
                parent, text=text, font=("Segoe UI", 13), text_color=COLORS["muted"]
            ).pack(anchor="w", pady=(0, 24))

    def welcome_page(self) -> None:
        # -------------------------------------------------------------
        # Welcome page background
        # -------------------------------------------------------------
        # Resolve assets from the application folder so the image also works
        # when the app is launched from a shortcut or another working folder.
        background_path = BASE_DIR / "assets" / "welcome_bg.jpg"

        if background_path.exists():
            self.content.update_idletasks()
            width = max(self.content.winfo_width(), 1180)
            height = max(self.content.winfo_height(), 800)
            source_image = Image.open(background_path).convert("RGB")
            # Contain the complete artwork instead of cropping its edges.
            # The navy canvas keeps the unused space intentional and polished.
            background_image = Image.new("RGB", (width, height), COLORS["navy"])
            contained = ImageOps.contain(
                source_image, (width, height), method=Image.Resampling.LANCZOS
            )
            offset = ((width - contained.width) // 2, (height - contained.height) // 2)
            background_image.paste(contained, offset)
            # A restrained navy overlay improves text contrast while keeping
            # the campus architecture and sky visibly present.
            overlay = Image.new("RGB", background_image.size, "#0B1F3A")
            background_image = Image.blend(background_image, overlay, 0.24)
            self._welcome_background_pil = background_image

            self._welcome_background_ctk = ctk.CTkImage(
                light_image=background_image,
                dark_image=background_image,
                size=(width, height),
            )
            self._welcome_background_label = ctk.CTkLabel(
                self.content,
                image=self._welcome_background_ctk,
                text="",
                fg_color="transparent",
            )
            self._welcome_background_label.place(
                relx=0, rely=0, relwidth=1, relheight=1
            )
            self._welcome_background_label.image = self._welcome_background_ctk
            self._welcome_resize_binding = self.content.bind(
                "<Configure>", self._resize_welcome_background, add="+"
            )
        else:
            LOGGER.warning("Welcome background not found: %s", background_path)

        # The content area is resizable, so keep the image crisp and fully
        # visible instead of leaving a fixed-size image with blank margins.
        self.content.after_idle(self._resize_welcome_background)

        # -------------------------------------------------------------
        # Main welcome content
        # -------------------------------------------------------------
        box = ctk.CTkFrame(
            self.content,
            fg_color="#F8FBFF",
            corner_radius=30,
            border_width=0,
        )
        box.place(relx=0.5, rely=0.50, anchor="center")

        # Replace the old (◉) symbol with the actual school logo.
        logo_path = BASE_DIR / "assets" / "school_logo.png"
        if logo_path.exists():
            logo_image = ctk.CTkImage(
                light_image=Image.open(logo_path).convert("RGBA"),
                dark_image=Image.open(logo_path).convert("RGBA"),
                size=(108, 108),
            )
            logo_label = ctk.CTkLabel(box, image=logo_image, text="")
            logo_label.pack(pady=(30, 10))
            logo_label.image = logo_image
        else:
            LOGGER.warning("School logo not found: %s", logo_path)
            ctk.CTkLabel(
                box,
                text="SMART ATTENDANCE",
                font=("Segoe UI", 16, "bold"),
                text_color=COLORS["blue"],
            ).pack(pady=(32, 12))

        ctk.CTkLabel(
            box,
            text="Face Recognition\nSmart Attendance System",
            justify="center",
            font=("Segoe UI", 31, "bold"),
            text_color=COLORS["navy"],
        ).pack(padx=64, pady=(0, 13))

        ctk.CTkLabel(
            box,
            text="Computer Engineering Fifth Year Project",
            font=("Segoe UI", 14, "bold"),
            text_color=COLORS["blue"],
        ).pack()

        ctk.CTkLabel(
            box,
            text="A secure, intelligent way to register students and manage attendance.",
            font=("Segoe UI", 13),
            text_color=COLORS["muted"],
        ).pack(padx=50, pady=(15, 30))

        ctk.CTkButton(
            box,
            text="Start  →",
            width=220,
            height=46,
            font=("Segoe UI", 15, "bold"),
            corner_radius=13,
            fg_color=COLORS["blue"],
            hover_color="#1D4ED8",
            command=lambda: self.show_page(1),
        ).pack(pady=(0, 8))

        ctk.CTkButton(
            box,
            text="Exit",
            width=220,
            height=38,
            corner_radius=13,
            fg_color="transparent",
            text_color=COLORS["muted"],
            hover_color="#E7EEF7",
            command=self.destroy,
        ).pack(pady=(0, 25))

    def _resize_welcome_background(self, _event=None) -> None:
        """Resize the welcome artwork to fill the available content area."""
        background = getattr(self, "_welcome_background_ctk", None)
        label = getattr(self, "_welcome_background_label", None)
        if background is None or label is None or not self.content.winfo_exists():
            return
        try:
            if not label.winfo_exists():
                return
            width = max(self.content.winfo_width(), 1)
            height = max(self.content.winfo_height(), 1)
            background.configure(size=(width, height))
        except tk.TclError:
            # The page may have been rebuilt while Tk still had a resize
            # callback queued for the previous welcome image.
            return

    def registration_page(self) -> None:
        self.page_heading(
            self.content,
            "New Student Registration",
            "Create the student profile before adding face data.",
        )
        tip = ctk.CTkFrame(self.content, corner_radius=14, fg_color=COLORS["soft_blue"])
        tip.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            tip,
            text="Start with the student’s ID and full name. Face samples are added in the next step.",
            font=("Segoe UI", 13),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=18, pady=15)
        card = self.card(self.content, width=600)
        card.pack(anchor="w", padx=8, pady=8, fill="x")
        ctk.CTkLabel(
            card,
            text="Student information",
            font=("Segoe UI", 17, "bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=30, pady=(25, 16))
        self.id_entry = ctk.CTkEntry(
            card, placeholder_text="Student ID (e.g. 1CE-001)", height=42
        )
        self.id_entry.pack(fill="x", padx=30, pady=7)
        self.name_entry = ctk.CTkEntry(card, placeholder_text="Full name", height=42)
        self.name_entry.pack(fill="x", padx=30, pady=7)
        self.registration_next = ctk.CTkButton(
            card,
            text="Next: Capture Face Data  →",
            state="disabled",
            command=lambda: self.show_page(2),
        )
        ctk.CTkButton(
            card, text="Save Student", height=42, command=self.save_student
        ).pack(fill="x", padx=30, pady=(18, 8))
        self.registration_next.pack(fill="x", padx=30, pady=(0, 25))
        ctk.CTkButton(
            self.content,
            text="Manage Registered Students",
            command=self.open_student_manager,
        ).pack(anchor="w", padx=8, pady=(10, 4))
        ctk.CTkButton(
            self.content,
            text="← Back",
            fg_color="transparent",
            text_color=COLORS["muted"],
            command=lambda: self.show_page(0),
        ).pack(anchor="w")

    def save_student(self) -> None:
        sid, name = self.id_entry.get().strip(), self.name_entry.get().strip()
        if not is_valid_student_id(sid) or not name:
            self.notify(
                "Enter a valid ID (letters, numbers, and hyphens, e.g. 1CE-001) and full name.",
                "error",
            )
            return
        if database.student_exists(sid):
            self.notify("This Student ID is already registered.", "error")
            return
        database.add_student(sid, name)
        self.student_id, self.student_name = sid, name
        self.training_required = True
        self.trained = False
        self.registration_next.configure(state="normal")
        self.notify(f"{name} was registered successfully.")

    def open_student_manager(self) -> None:
        """Open the edit/delete control for registered student data."""
        students = database.list_students()
        if not students:
            self.notify("No students are registered yet.", "warning")
            return
        dialog = ctk.CTkToplevel(self)
        dialog.title("Manage Student Information")
        dialog.geometry("560x430")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        choices = [f"{student_id} — {name}" for student_id, name in students]
        selected = tk.StringVar(value=choices[0])
        ctk.CTkLabel(
            dialog,
            text="Edit or delete a registered student",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", padx=30, pady=(28, 16))
        selector = ctk.CTkComboBox(dialog, variable=selected, values=choices, width=460)
        selector.pack(fill="x", padx=30, pady=(0, 16))
        id_entry = ctk.CTkEntry(dialog, height=40)
        name_entry = ctk.CTkEntry(dialog, height=40)

        def selected_id() -> str:
            return selected.get().split(" — ", 1)[0].strip()

        def load_selected(_: str | None = None) -> None:
            record = database.get_student(selected_id())
            if record:
                id_entry.delete(0, "end")
                id_entry.insert(0, record[0])
                name_entry.delete(0, "end")
                name_entry.insert(0, record[1])

        selector.configure(command=load_selected)
        ctk.CTkLabel(dialog, text="Student ID").pack(anchor="w", padx=30)
        id_entry.pack(fill="x", padx=30, pady=(4, 12))
        ctk.CTkLabel(dialog, text="Student name").pack(anchor="w", padx=30)
        name_entry.pack(fill="x", padx=30, pady=(4, 18))
        load_selected()

        def save_changes() -> None:
            old_id, new_id, new_name = (
                selected_id(),
                id_entry.get().strip(),
                name_entry.get().strip(),
            )
            if not is_valid_student_id(new_id) or not new_name:
                messagebox.showerror(
                    "Invalid information",
                    "Enter a valid Student ID and name.",
                    parent=dialog,
                )
                return
            if old_id != new_id and (DATASET_DIR / new_id).exists():
                messagebox.showerror(
                    "Update failed",
                    "A face-data folder already exists for that Student ID.",
                    parent=dialog,
                )
                return
            try:
                database.update_student(old_id, new_id, new_name)
                reset_database.move_student_dataset(old_id, new_id)
            except Exception as error:
                messagebox.showerror("Update failed", str(error), parent=dialog)
                return
            self.student_id, self.student_name, self.trained = new_id, new_name, False
            self.training_required = True
            self.recognition_cooldowns.clear()
            dialog.destroy()
            self.notify(
                "Student information updated. Retrain the model before live attendance.",
                "warning",
            )
            self.show_page(1)

        def remove_student() -> None:
            old_id = selected_id()
            if not messagebox.askyesno(
                "Delete student",
                f"Permanently delete {old_id}, their attendance records, face images, and trained model data?",
                parent=dialog,
            ):
                return
            try:
                reset_database.delete_student_data(old_id)
            except Exception as error:
                messagebox.showerror("Delete failed", str(error), parent=dialog)
                return
            if self.student_id == old_id:
                self.student_id = self.student_name = ""
            self.trained = False
            self.training_required = bool(database.list_students())
            self.recognition_cooldowns.pop(old_id, None)
            dialog.destroy()
            self.notify(
                "Student data was deleted. Retrain the model for remaining students.",
                "warning",
            )
            self.show_page(1)

        actions = ctk.CTkFrame(dialog, fg_color="transparent")
        actions.pack(fill="x", padx=30)
        ctk.CTkButton(
            actions,
            text="Delete Student",
            fg_color=COLORS["red"],
            command=remove_student,
        ).pack(side="left")
        ctk.CTkButton(actions, text="Save Changes", command=save_changes).pack(
            side="right"
        )

    def capture_page(self) -> None:
        scroll = ctk.CTkScrollableFrame(
            self.content,
            fg_color="transparent",
            scrollbar_button_color="#B8C7D9",
            scrollbar_button_hover_color=COLORS["blue"],
        )
        scroll.pack(fill="both", expand=True)
        self.page_heading(
            scroll,
            "Capture Face Dataset",
            "Collect clear face samples for the recognition model.",
        )

        top = ctk.CTkFrame(scroll, corner_radius=14, fg_color=COLORS["soft_blue"])
        top.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            top,
            text=f"Student:  {self.student_name or 'Select a student'}    •    ID: {self.student_id or '—'}",
            font=("Segoe UI", 14, "bold"),
            text_color=COLORS["blue"],
        ).pack(anchor="w")

        self.capture_progress = ctk.CTkProgressBar(scroll)
        self.capture_progress.set(0)
        self.capture_progress.pack(fill="x", pady=(8, 6))
        self.capture_text = ctk.CTkLabel(
            scroll,
            text=f"0 / {recognition_settings()['samples']} face samples",
            text_color=COLORS["muted"],
        )
        self.capture_text.pack(anchor="w", pady=(0, 12))

        row = ctk.CTkFrame(scroll, fg_color="transparent")
        self.capture_scroll = scroll
        row.pack(fill="x", pady=(0, 16))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)

        card = self.card(row)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ctk.CTkLabel(card, text="📷", font=("Segoe UI Emoji", 35)).pack(
            anchor="w", padx=24, pady=(25, 8)
        )
        ctk.CTkLabel(card, text="Camera Capture", font=("Segoe UI", 18, "bold")).pack(
            anchor="w", padx=24
        )
        ctk.CTkLabel(
            card,
            text="Capture guided face angles: front, left, right, up, and down.",
            wraplength=320,
            justify="left",
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=24, pady=10)
        ctk.CTkButton(
            card, text="Start Camera Capture", command=self.capture_camera
        ).pack(anchor="w", padx=24, pady=(8, 25))

        update_card = self.card(row)
        update_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        ctk.CTkLabel(update_card, text="🛠️", font=("Segoe UI Emoji", 28)).pack(
            anchor="w", padx=24, pady=(18, 6)
        )
        ctk.CTkLabel(
            update_card, text="Update Existing Face Data", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w", padx=24)
        ctk.CTkLabel(
            update_card,
            text="Add more photos for a registered student to sharpen recognition accuracy.",
            wraplength=320,
            justify="left",
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=24, pady=(4, 16))
        ctk.CTkButton(
            update_card,
            text="Update Face Data",
            command=self.open_update_face_data_dialog,
        ).pack(anchor="w", padx=24, pady=(0, 20))

        self._add_training_action(scroll)
        ctk.CTkButton(
            scroll,
            text="Continue to Live Attendance  →",
            command=lambda: self.show_page(3),
        ).pack(anchor="e", pady=(10, 4))

    def _check_capture_student(self) -> bool:
        if self.student_id:
            return True
        self.notify("Register a student before adding face data.", "error")
        return False

    def capture_camera(self) -> None:
        if not self._check_capture_student():
            return
        if getattr(self, "capture_preview_window", None) is not None:
            self.notify("Face capture is already running.", "warning")
            return
        self.capture_stop.clear()
        self._open_capture_preview()
        threading.Thread(target=self._camera_capture_worker, daemon=True).start()

    def _open_capture_preview(self) -> None:
        """Show an in-app live preview while a student positions their face."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Face Sample Capture")
        dialog.geometry("840x660")
        dialog.minsize(720, 570)
        dialog.transient(self)
        self.capture_preview_window = dialog
        ctk.CTkLabel(
            dialog, text="Live Face Capture", font=("Segoe UI", 22, "bold")
        ).pack(pady=(22, 6))
        ctk.CTkLabel(
            dialog,
            text="Center your face in the frame and slowly change your angle. Samples are saved automatically.",
            text_color=COLORS["muted"],
        ).pack(pady=(0, 12))
        self.capture_preview_label = ctk.CTkLabel(
            dialog,
            text="Starting camera…",
            width=760,
            height=430,
            fg_color="#101828",
            text_color="white",
        )
        self.capture_preview_label.pack(padx=28, pady=8, fill="both", expand=True)
        self.capture_preview_status = ctk.CTkLabel(
            dialog,
            text="Preparing camera…",
            text_color=COLORS["blue"],
            font=("Segoe UI", 13, "bold"),
        )
        self.capture_preview_status.pack(pady=(8, 12))

        def stop_capture() -> None:
            self.capture_stop.set()
            self.capture_preview_status.configure(text="Stopping capture…")

        ctk.CTkButton(
            dialog, text="Stop Capture", fg_color=COLORS["red"], command=stop_capture
        ).pack(pady=(0, 22))
        dialog.protocol("WM_DELETE_WINDOW", stop_capture)

    def _update_capture_preview(
        self, image: Image.Image, count: int, target: int
    ) -> None:
        dialog = getattr(self, "capture_preview_window", None)
        if dialog is None or not dialog.winfo_exists():
            return
        photo = ctk.CTkImage(light_image=image, size=(760, 430))
        self.capture_preview_label.configure(image=photo, text="")
        self.capture_preview_label.image = photo
        self.capture_preview_status.configure(
            text=f"Saved {count} of {target} face samples"
        )

    def _close_capture_preview(self) -> None:
        dialog = getattr(self, "capture_preview_window", None)
        if dialog is not None:
            try:
                if dialog.winfo_exists():
                    dialog.destroy()
            except tk.TclError:
                pass
        self.capture_preview_window = None

    def _camera_capture_worker(self) -> None:
        target = recognition_settings()["samples"]
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            self.after(0, lambda: self.notify("Could not access the webcam.", "error"))
            self.after(0, self._close_capture_preview)
            return
        detector = recognition._detector()
        directory = DATASET_DIR / self.student_id
        directory.mkdir(parents=True, exist_ok=True)
        count = len(
            [
                path
                for path in directory.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        try:
            while count < target and not self.capture_stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera did not return a frame.")
                faces = recognition.detect_faces(frame, detector)
                preview = frame.copy()
                for x, y, width, height in faces:
                    cv2.rectangle(
                        preview, (x, y), (x + width, y + height), (0, 180, 0), 2
                    )
                cv2.putText(
                    preview,
                    f"Capturing {count}/{target}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 180, 0),
                    2,
                )
                display_image = Image.fromarray(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                )
                self.after(
                    0,
                    lambda image=display_image, current=count, total=target: self._update_capture_preview(
                        image, current, total
                    ),
                )
                if faces:
                    try:
                        face = recognition.prepare_face(
                            frame, tuple(max(faces, key=lambda r: r[2] * r[3]))
                        )
                        output_name = f"{count:03d}.jpg"
                        cv2.imwrite(str(directory / output_name), face)
                        count += 1
                        self.after(0, lambda n=count: self._set_capture_progress(n))
                    except ValueError:
                        LOGGER.debug(
                            "Rejected live capture frame due to preprocessing quality."
                        )
        except Exception as error:
            self.after(0, lambda: self.notify(str(error), "error"))
        finally:
            camera.release()
            self.after(0, self._close_capture_preview)
            self.after(0, lambda: self._dataset_complete(count))

    def _set_capture_progress(self, count: int) -> None:
        if hasattr(self, "capture_progress"):
            target = recognition_settings()["samples"]
            self.capture_progress.set(min(1, count / target))
            self.capture_text.configure(text=f"{count} / {target} face samples")

    def _dataset_complete(self, count: int) -> None:
        self.trained = False
        self.training_required = True
        self.notify(
            f"Saved {count} samples. Retrain the model before live attendance.",
            "warning",
        )

    def upload_photos(self) -> None:
        if not self._check_capture_student():
            return
        files = filedialog.askopenfilenames(
            title="Select student photos",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")],
        )
        if files:
            threading.Thread(
                target=self._upload_worker, args=(files,), daemon=True
            ).start()

    def _upload_worker(self, files) -> None:
        detector = recognition._detector()
        folder = DATASET_DIR / self.student_id
        folder.mkdir(parents=True, exist_ok=True)
        count = len(
            [
                path
                for path in folder.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        for path in files:
            image = cv2.imread(path)
            if image is None:
                LOGGER.warning("Skipping unreadable upload file: %s", path)
                continue
            faces = recognition.detect_faces(image, detector)
            if not faces:
                LOGGER.warning("No face found in upload file: %s", path)
                continue
            try:
                face = recognition.prepare_face(
                    image, tuple(max(faces, key=lambda r: r[2] * r[3]))
                )
                output_name = f"{count:03d}.jpg"
                cv2.imwrite(str(folder / output_name), face)
                count += 1
                self.after(0, lambda n=count: self._set_capture_progress(n))
            except ValueError as error:
                LOGGER.warning("Uploaded image rejected: %s (%s)", path, error)
                continue
        self.after(0, lambda: self._dataset_complete(count))

    def open_update_face_data_dialog(self) -> None:
        students = database.list_students()
        if not students:
            self.notify("No students are registered yet.", "error")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Update Face Data")
        dialog.geometry("760x620")
        dialog.resizable(False, False)
        dialog.configure(fg_color=COLORS["bg"])
        dialog.transient(self)
        dialog.grab_set()
        self.update_face_dialog = dialog

        student_options = [f"{student_id} — {name}" for student_id, name in students]
        student_var = tk.StringVar(value=student_options[0])
        search_var = tk.StringVar(value="")
        self.update_face_status_var = tk.StringVar(
            value="Select a student and choose how to add more face samples."
        )
        self.update_face_summary_var = tk.StringVar(value="")
        self.update_face_progress = ctk.CTkProgressBar(dialog)
        self.update_face_progress.set(0)
        # A normal CTk layout is reliable across Windows themes. The previous
        # Tk Canvas/CTk embedding could render this dialog as a black window.
        content = ctk.CTkFrame(dialog, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=28, pady=(20, 18))

        ctk.CTkLabel(
            content,
            text="Update Existing Student Face Data",
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ctk.CTkLabel(
            content,
            text="Choose a registered student and add new face samples.",
            text_color=COLORS["muted"],
        ).pack(anchor="w", pady=(0, 14))

        search_frame = ctk.CTkFrame(content, fg_color="transparent")
        search_frame.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(search_frame, text="Search student ID").pack(anchor="w")
        search_entry = ctk.CTkEntry(search_frame, textvariable=search_var)
        search_entry.pack(fill="x", pady=(4, 8))

        def filter_students() -> None:
            query = search_var.get().strip().lower()
            filtered = [option for option in student_options if query in option.lower()]
            combo.configure(values=filtered or student_options)
            if filtered:
                combo.set(filtered[0])
            else:
                combo.set("")

        ctk.CTkButton(search_frame, text="Find Student", command=filter_students).pack(
            anchor="w"
        )
        combo = ctk.CTkComboBox(
            content, variable=student_var, values=student_options, width=360
        )
        combo.pack(fill="x", pady=(4, 10))

        button_row = ctk.CTkFrame(content, fg_color="transparent")
        button_row.pack(fill="x", pady=(6, 8))

        def resolve_student_id() -> str:
            value = student_var.get().strip()
            if not value:
                raise ValueError("Please select a registered student.")
            student_id = value.split(" — ", 1)[0]
            if not database.student_exists(student_id):
                raise ValueError("Student not found.")
            return student_id

        def apply_upload() -> None:
            try:
                student_id = resolve_student_id()
            except ValueError as error:
                self.notify(str(error), "error")
                return
            files = filedialog.askopenfilenames(
                title="Select additional student photos",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")],
            )
            if not files:
                self.notify("No images were selected.", "warning")
                return
            self._start_update_face_import(
                dialog, student_id, list(files), method="upload"
            )

        def apply_camera() -> None:
            try:
                student_id = resolve_student_id()
            except ValueError as error:
                self.notify(str(error), "error")
                return
            if getattr(self, "capture_preview_window", None) is not None:
                self.notify("Face capture is already running.", "warning")
                return
            self.capture_stop.clear()
            self._open_capture_preview()
            self.capture_preview_window.title("Update Existing Student Face Data")
            # Keep the capture experience focused: the configuration dialog is
            # restored only after the live webcam capture has finished.
            dialog.grab_release()
            dialog.withdraw()
            self._start_update_face_import(dialog, student_id, [], method="camera")

        ctk.CTkButton(button_row, text="Select Photos", command=apply_upload).pack(
            side="left", padx=(0, 10)
        )
        ctk.CTkButton(
            button_row, text="Start Webcam Capture", command=apply_camera
        ).pack(side="left")

        self.update_face_status = ctk.CTkLabel(
            content,
            textvariable=self.update_face_status_var,
            text_color=COLORS["muted"],
            wraplength=700,
        )
        self.update_face_progress.pack(fill="x", pady=(6, 8))
        self.update_face_status.pack(anchor="w", pady=(2, 8))
        self.update_face_summary = ctk.CTkLabel(
            content,
            textvariable=self.update_face_summary_var,
            justify="left",
            text_color=COLORS["navy"],
            font=("Segoe UI", 12),
        )
        self.update_face_summary.pack(anchor="w", pady=(0, 18))
        ctk.CTkButton(content, text="Close", command=dialog.destroy).pack(anchor="e")

    def _set_update_progress(self, current: int, total: int) -> None:
        if not hasattr(self, "update_face_progress"):
            return
        if total <= 0:
            self.update_face_progress.set(0)
            return
        self.update_face_progress.set(min(1.0, current / total))
        self.update_face_status_var.set(f"Processing {current}/{total} image(s)...")

    def _start_update_face_import(
        self, dialog: ctk.CTkToplevel, student_id: str, files: list[str], method: str
    ) -> None:
        self.update_face_status_var.set("Preparing to add face samples...")
        self.update_face_progress.set(0)
        self.update_face_progress.start()
        self.update_face_summary_var.set("")

        def worker() -> None:
            try:
                if method == "camera":
                    stats = self._capture_update_face_samples(student_id)
                else:
                    stats = recognition.add_face_samples_for_student(
                        student_id,
                        files,
                        progress_callback=lambda current, total: self.after(
                            0,
                            lambda c=current, t=total: self._set_update_progress(c, t),
                        ),
                    )
                self.after(
                    0,
                    lambda: self._finish_update_face_import(dialog, student_id, stats),
                )
            except Exception as error:
                if method == "camera":
                    self.after(0, self._close_capture_preview)
                self.after(
                    0,
                    lambda caught_error=error: self._fail_update_face_import(
                        dialog, caught_error
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _capture_update_face_samples(self, student_id: str) -> dict[str, int]:
        folder = DATASET_DIR / student_id
        folder.mkdir(parents=True, exist_ok=True)
        detector = recognition._detector()
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            raise RuntimeError("Could not access the webcam.")
        target_count = recognition_settings()["samples"]
        stats = {
            "images_selected": 0,
            "valid_faces": 0,
            "ignored_images": 0,
            "new_images_added": 0,
        }
        try:
            for index in range(target_count):
                if self.capture_stop.is_set():
                    break
                ok, frame = camera.read()
                if not ok:
                    break
                faces = recognition.detect_faces(frame, detector)
                preview = frame.copy()
                for x, y, width, height in faces:
                    cv2.rectangle(
                        preview, (x, y), (x + width, y + height), (0, 180, 0), 2
                    )
                cv2.putText(
                    preview,
                    f"Updating face data {stats['new_images_added']}/{target_count}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 180, 0),
                    2,
                )
                display_image = Image.fromarray(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                )
                self.after(
                    0,
                    lambda image=display_image, current=stats[
                        "new_images_added"
                    ], total=target_count: self._update_capture_preview(
                        image, current, total
                    ),
                )
                if not faces:
                    stats["ignored_images"] += 1
                    continue
                try:
                    face = recognition.prepare_face(
                        frame, tuple(max(faces, key=lambda rect: rect[2] * rect[3]))
                    )
                except ValueError:
                    stats["ignored_images"] += 1
                    continue
                output_path = recognition._unique_dataset_path(
                    folder, Path(f"capture{index + 1}.jpg")
                )
                saved = cv2.imwrite(str(output_path), face)
                if not saved:
                    stats["ignored_images"] += 1
                    continue
                stats["images_selected"] += 1
                stats["valid_faces"] += 1
                stats["new_images_added"] += 1
                self.after(
                    0,
                    lambda current=index + 1, total=target_count: self._set_update_progress(
                        current, total
                    ),
                )
        finally:
            camera.release()
            self.after(0, self._close_capture_preview)
        return stats

    def _finish_update_face_import(
        self, dialog: ctk.CTkToplevel, student_id: str, stats: dict[str, int]
    ) -> None:
        if dialog.winfo_exists():
            dialog.deiconify()
            dialog.grab_set()
            self.update_face_progress.stop()
            self.update_face_progress.set(1)
            self.update_face_status_var.set("Face samples imported successfully.")
            self._last_update_stats = stats
            self._last_update_student_id = student_id
            self.update_face_summary_var.set(
                "\n".join(
                    [
                        f"Images Selected: {stats['images_selected']}",
                        f"Valid Faces: {stats['valid_faces']}",
                        f"Ignored Images: {stats['ignored_images']}",
                        f"New Images Added: {stats['new_images_added']}",
                        "Training Time: pending",
                    ]
                )
            )
            self.notify(
                f"Added {stats['new_images_added']} new images for {student_id}."
            )
            self.trained = False
            self.training_required = True
            if self.current_page == 2 and hasattr(self, "capture_scroll"):
                self._add_training_action(self.capture_scroll)
            self.update_face_summary_var.set(
                "\n".join(
                    [
                        f"Images Selected: {stats['images_selected']}",
                        f"Valid Faces: {stats['valid_faces']}",
                        f"Ignored Images: {stats['ignored_images']}",
                        f"New Images Added: {stats['new_images_added']}",
                        "Training Time: not run",
                    ]
                )
            )
            self.update_face_status_var.set(
                "New images have been saved. Start training manually when ready."
            )
            self.notify(
                "New images have been saved. Start training manually when ready.",
                "warning",
            )

    def _start_update_training(self, dialog: ctk.CTkToplevel, student_id: str) -> None:
        self.update_face_status_var.set(
            "Training recognition model with all available face samples..."
        )
        self.update_face_progress.set(0)
        self.update_face_progress.start()

        def worker() -> None:
            start_time = time.perf_counter()
            try:
                samples, students = recognition.train_model()
                elapsed = time.perf_counter() - start_time
                self.after(
                    0,
                    lambda: self._finish_update_training(
                        dialog, student_id, samples, students, elapsed
                    ),
                )
            except Exception as error:
                self.after(
                    0,
                    lambda caught_error=error: self._fail_update_training(
                        dialog, caught_error
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_update_training(
        self,
        dialog: ctk.CTkToplevel,
        student_id: str,
        samples: int,
        students: int,
        elapsed: float,
    ) -> None:
        if dialog.winfo_exists():
            self.trained = True
            self.training_required = False
            self.update_face_progress.stop()
            self.update_face_progress.set(1)
            self.update_face_status_var.set("Training complete.")
            stats = getattr(
                self,
                "_last_update_stats",
                {
                    "images_selected": 0,
                    "valid_faces": 0,
                    "ignored_images": 0,
                    "new_images_added": 0,
                },
            )
            self.update_face_summary_var.set(
                "\n".join(
                    [
                        f"Images Selected: {stats['images_selected']}",
                        f"Valid Faces: {stats['valid_faces']}",
                        f"Ignored Images: {stats['ignored_images']}",
                        f"New Images Added: {stats['new_images_added']}",
                        f"Training Time: {elapsed:.2f}s",
                    ]
                )
            )
            self.notify(
                f"Training complete — {samples} samples from {students} student(s) for {student_id}."
            )

    def _fail_update_face_import(
        self, dialog: ctk.CTkToplevel, error: Exception
    ) -> None:
        if dialog.winfo_exists():
            dialog.deiconify()
            dialog.grab_set()
            self.update_face_progress.stop()
            self.update_face_progress.set(0)
            self.update_face_status_var.set(str(error))
            self.update_face_summary_var.set(
                "Import failed. Check the selected files and student folder."
            )
            self.notify(str(error), "error")

    def _fail_update_training(self, dialog: ctk.CTkToplevel, error: Exception) -> None:
        if dialog.winfo_exists():
            self.update_face_progress.stop()
            self.update_face_progress.set(0)
            self.update_face_status_var.set(str(error))
            self.update_face_summary_var.set(
                "Training failed. Please verify the dataset and try again."
            )
            self.notify(str(error), "error")

    def _add_training_action(self, parent) -> None:
        """Keep model training contextual instead of using a separate page."""
        if not self.training_required:
            return
        existing = getattr(self, "training_action_card", None)
        if existing is not None and existing.winfo_exists():
            return
        card = self.card(parent)
        self.training_action_card = card
        card.pack(fill="x", pady=(18, 12))
        ctk.CTkLabel(
            card,
            text="Training required",
            font=("Segoe UI", 17, "bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=24, pady=(18, 3))
        self.training_status = ctk.CTkLabel(
            card,
            text="Face data changed. Train the model before using live attendance.",
            text_color=COLORS["muted"],
        )
        self.training_status.pack(anchor="w", padx=24, pady=(0, 10))
        self.training_bar = ctk.CTkProgressBar(card)
        self.training_bar.set(0)
        self.training_bar.pack(fill="x", padx=24, pady=(0, 10))
        self.train_button = ctk.CTkButton(
            card, text="Train Model", height=40, command=self.train_model
        )
        self.train_button.pack(anchor="w", padx=24, pady=(0, 18))
        self.train_next = self.train_button

    def train_model(self) -> None:
        self.train_button.configure(state="disabled")
        self.training_bar.start()
        self.training_status.configure(
            text="Analysing samples and training recognizer…"
        )
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        try:
            samples, students = recognition.train_model()
            self.after(
                0,
                lambda: self._training_done(
                    f"Training complete — {samples} samples from {students} student(s)."
                ),
            )
        except Exception as error:
            self.after(0, lambda: self._training_failed(str(error)))

    def _training_done(self, message: str) -> None:
        self.training_required = False
        self.training_bar.stop()
        self.training_bar.set(1)
        self.trained = True
        self.train_next.configure(state="normal")
        self.training_status.configure(text=message)
        self.notify(message)

    def _training_failed(self, message: str) -> None:
        self.training_bar.stop()
        self.train_button.configure(state="normal")
        self.training_status.configure(text=message)
        self.notify(message, "error")

    def attendance_page(self) -> None:
        self.live_attendance_records = []
        scroll = ctk.CTkScrollableFrame(
            self.content,
            fg_color="transparent",
            scrollbar_button_color="#B8C7D9",
            scrollbar_button_hover_color=COLORS["blue"],
        )
        scroll.pack(fill="both", expand=True)
        self.page_heading(
            scroll,
            "Live Attendance",
            "New scans appear only for this live session; each student is recorded once per scheduled time period.",
        )

        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x")
        left = self.card(row)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.video_label = ctk.CTkLabel(
            left,
            text="Camera preview will appear here",
            width=600,
            height=380,
            font=("Segoe UI", 16),
            text_color=COLORS["muted"],
            anchor="center",
        )
        self.video_label.pack(fill="both", expand=True, padx=15, pady=15)

        right = self.card(row, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ctk.CTkLabel(
            right, text="Recognition Result", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w", padx=25, pady=(28, 15))
        self.result_name = ctk.CTkLabel(
            right,
            text="Waiting for camera",
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS["blue"],
            justify="left",
            wraplength=300,
            width=28,
        )
        self.result_name.pack(anchor="w", padx=25, fill="x")
        self.result_details = ctk.CTkLabel(
            right,
            text="Student ID: —\nTime: —\nConfidence: —",
            justify="left",
            text_color=COLORS["muted"],
            font=("Segoe UI", 13),
            wraplength=300,
            width=28,
        )
        self.result_details.pack(anchor="w", padx=25, pady=14, fill="x")
        ctk.CTkButton(right, text="Start Camera", command=self.start_camera).pack(
            fill="x", padx=25, pady=(15, 8)
        )
        ctk.CTkButton(
            right,
            text="Stop Attendance",
            fg_color=COLORS["red"],
            command=self.stop_camera,
        ).pack(fill="x", padx=25)

        table = self.card(scroll)
        table.pack(fill="both", expand=True, pady=15)
        table.grid_columnconfigure(0, weight=1)
        table.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            table, text="Current Live-Session Attendance", font=("Segoe UI", 16, "bold")
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(20, 12))
        table_body = ctk.CTkFrame(table, fg_color="transparent")
        table_body.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 22))
        table_body.grid_columnconfigure(0, weight=1)
        table_body.grid_rowconfigure(0, weight=1)
        self.attendance_rows = ctk.CTkTextbox(
            table_body, height=390, font=("Consolas", 13), wrap="none"
        )
        self.attendance_rows.grid(row=0, column=0, sticky="nsew")
        self.refresh_attendance_table()

        ctk.CTkButton(
            scroll, text="Continue to Reports  →", command=lambda: self.show_page(4)
        ).pack(anchor="e", pady=(10, 4))

    def start_camera(self) -> None:
        if not recognition.model_available():
            self.notify("Train a model before starting live attendance.", "error")
            return
        self.camera_stop.clear()
        threading.Thread(target=self._recognition_worker, daemon=True).start()
        self.after(30, self._poll_camera)

    def stop_camera(self) -> None:
        self.camera_stop.set()

    def _recognition_worker(self) -> None:
        settings = recognition_settings()
        recognizer = recognition._load_sface()
        label_mapping = recognition._load_label_mapping()
        detector = recognition._detector()
        camera = cv2.VideoCapture(CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            self.after(0, lambda: self.notify("Could not access the webcam.", "error"))
            return
        try:
            while not self.camera_stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    break
                event = None
                for rect in recognition.detect_faces(frame, detector):
                    x, y, w, h = map(int, rect)
                    try:
                        face = recognition.prepare_face(frame, (x, y, w, h))
                    except ValueError:
                        LOGGER.debug(
                            "Rejected live frame candidate due to preprocessing quality."
                        )
                        continue
                    sid, confidence = recognition.predict_face(
                        face, recognizer, label_mapping, settings["threshold"]
                    )
                    session = database.attendance_session()
                    if (
                        sid
                        and session
                        and time.monotonic() - self.recognition_cooldowns.get(sid, 0)
                        >= settings["cooldown"]
                    ):
                        added = database.mark_attendance(sid)
                        self.recognition_cooldowns[sid] = time.monotonic()
                    else:
                        added = False
                    student_record = database.get_student(sid) if sid else None
                    name = student_record[1] if student_record else "Unknown"
                    color = (0, 180, 0) if sid else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        f"{name} ({confidence:.0f})",
                        (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                    )
                    recognition.log_recognition(
                        sid or "", name, confidence, "Recognized" if sid else "Unknown"
                    )
                    event = (sid, name, confidence, added, session)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not self.camera_queue.full():
                    self.camera_queue.put((image, event))
        except Exception as error:
            LOGGER.exception("Recognition failed")
            self.after(0, lambda: self.notify(str(error), "error"))
        finally:
            camera.release()

    def _poll_camera(self) -> None:
        try:
            image, event = self.camera_queue.get_nowait()
            photo = ctk.CTkImage(light_image=image, size=(600, 370))
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            if event:
                sid, name, confidence, added, session = event
                now = datetime.now().strftime("%H:%M:%S")
                self.result_name.configure(
                    text=name, text_color=COLORS["green"] if sid else COLORS["red"]
                )
                self.result_details.configure(
                    text=f"Student ID: {sid or 'Unknown'}\nTime: {now}\nConfidence: {confidence:.1f}"
                )
                if added:
                    self.live_attendance_records.insert(
                        0, (now, session or "", sid or "", name)
                    )
                    self.notify(f"Attendance recorded for {name} ({session}).")
                    self.refresh_attendance_table()
        except queue.Empty:
            pass
        if not self.camera_stop.is_set() and self.current_page == 3:
            self.after(30, self._poll_camera)

    def refresh_attendance_table(self) -> None:
        if not hasattr(self, "attendance_rows"):
            return
        self.attendance_rows.configure(state="normal")
        self.attendance_rows.delete("1.0", "end")
        self.attendance_rows.insert(
            "end",
            "TIME       SESSION          STUDENT ID          NAME\n" + "─" * 70 + "\n",
        )
        for scan_time, session, sid, name in self.live_attendance_records:
            self.attendance_rows.insert(
                "end", f"{scan_time:<10} {session:<16} {sid:<19} {name}\n"
            )
        self.attendance_rows.configure(state="disabled")

    def reports_page(self) -> None:
        self.page_heading(
            self.content,
            "Reports & Export",
            "Review system activity and share attendance records.",
        )
        metrics = database.dashboard_summary()
        metrics["accuracy"] = (
            "Available" if recognition.model_available() else "Not trained"
        )
        row = ctk.CTkFrame(self.content, fg_color="transparent")
        row.pack(fill="x")
        for label, value, icon in [
            ("Registered Students", metrics["students"], "👤"),
            ("Today’s Attendance", metrics["today"], "✓"),
            ("Today’s Total Records", metrics["records"], "▣"),
            ("Recognition Model", metrics["accuracy"], "◉"),
        ]:
            card = self.card(row)
            card.pack(side="left", expand=True, fill="x", padx=6)
            ctk.CTkLabel(
                card, text=icon, font=("Segoe UI", 24), text_color=COLORS["blue"]
            ).pack(anchor="w", padx=18, pady=(18, 3))
            ctk.CTkLabel(
                card,
                text=str(value),
                font=("Segoe UI", 22, "bold"),
                text_color=COLORS["text"],
            ).pack(anchor="w", padx=18)
            ctk.CTkLabel(card, text=label, text_color=COLORS["muted"]).pack(
                anchor="w", padx=18, pady=(0, 18)
            )
        actions = self.card(self.content)
        actions.pack(fill="x", pady=25)
        ctk.CTkLabel(
            actions, text="Export & delivery", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w", padx=25, pady=(22, 12))
        for text, command, color in [
            ("📄  Export CSV", self.export_report, COLORS["blue"]),
            ("📂  Open Folder", self.open_exports, COLORS["navy"]),
            ("📧  Send Report", self.open_email_dialog, COLORS["green"]),
        ]:
            ctk.CTkButton(
                actions,
                text=text,
                width=240,
                height=40,
                fg_color=color,
                command=command,
            ).pack(side="left", padx=(25, 0), pady=(0, 25))

        reset_card = self.card(self.content)
        reset_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            reset_card,
            text="Remove all registered students, attendance history, trained model data, labels, and face-image samples.",
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=25, pady=(10, 14))
        ctk.CTkButton(
            reset_card,
            text="🗑️  Delete All Registered Data",
            width=280,
            height=42,
            fg_color=COLORS["red"],
            command=self.reset_all_data,
        ).pack(anchor="w", padx=25, pady=(0, 22))

    def reset_all_data(self) -> None:
        if not messagebox.askyesno(
            "Reset all data",
            "This will permanently delete all students, attendance records, trained model files, labels, and dataset images. Continue?",
        ):
            return
        try:
            reset_database.reset_all_data()
            self.student_id = ""
            self.student_name = ""
            self.trained = False
            self.training_required = False
            self.recognition_cooldowns.clear()
            self.live_attendance_records = []
            self.notify("All registered data was deleted.")
            self.show_page(0)
        except Exception as error:
            self.notify(str(error), "error")

    def export_report(self) -> Path | None:
        try:
            report = database.export_to_csv()
            self.notify(f"Report exported to {report.name}.")
            return report
        except Exception as error:
            self.notify(str(error), "error")
            return None

    def open_exports(self) -> None:
        ensure_runtime_directories()
        try:
            import os

            if hasattr(os, "startfile"):
                os.startfile(str(EXPORTS_DIR))
            else:
                import subprocess

                subprocess.run(["xdg-open", str(EXPORTS_DIR)], check=False)
        except Exception as error:
            self.notify(str(error), "error")

    def open_email_dialog(self) -> None:
        """Open the email form. Credentials remain only in this dialog's memory."""
        settings = get_settings()
        dialog = ctk.CTkToplevel(self)
        dialog.title("Email Attendance Report")
        dialog.geometry("620x790")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        fields = {}
        saved_recipients = settings.get("teacher_emails", [])
        if not isinstance(saved_recipients, list):
            saved_recipients = []
        recipient_choices = list(
            dict.fromkeys(
                [
                    email
                    for email in [settings.get("teacher_email", ""), *saved_recipients]
                    if email
                ]
            )
        )
        ctk.CTkLabel(dialog, text="Teacher Email").pack(
            anchor="w", padx=30, pady=(13, 2)
        )
        fields["recipient"] = ctk.CTkComboBox(
            dialog, values=recipient_choices or ["Select a saved email"], height=32
        )
        if recipient_choices:
            fields["recipient"].set(recipient_choices[0])
        fields["recipient"].pack(fill="x", padx=30)
        email_actions = ctk.CTkFrame(dialog, fg_color="transparent")
        email_actions.pack(fill="x", padx=30, pady=(8, 0))

        def add_teacher_email() -> None:
            email = fields["recipient"].get().strip()
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                messagebox.showerror(
                    "Invalid email",
                    "Enter a valid teacher email address before adding it.",
                    parent=dialog,
                )
                return
            values = list(dict.fromkeys([email, *fields["recipient"].cget("values")]))
            values = [value for value in values if value != "Select a saved email"]
            fields["recipient"].configure(values=values)
            fields["recipient"].set(email)
            save_settings({"teacher_email": email, "teacher_emails": values})
            self.notify("Teacher email address saved.")

        def delete_teacher_email() -> None:
            email = fields["recipient"].get().strip()
            values = [
                value
                for value in fields["recipient"].cget("values")
                if value not in (email, "Select a saved email")
            ]
            if email not in fields["recipient"].cget("values"):
                messagebox.showinfo(
                    "Select an email",
                    "Select a saved teacher email address to delete.",
                    parent=dialog,
                )
                return
            fields["recipient"].configure(values=values or ["Select a saved email"])
            fields["recipient"].set(values[0] if values else "Select a saved email")
            save_settings(
                {"teacher_email": values[0] if values else "", "teacher_emails": values}
            )
            self.notify("Teacher email address removed.", "warning")

        ctk.CTkButton(
            email_actions, text="+ Add", width=100, height=30, command=add_teacher_email
        ).pack(side="left")
        ctk.CTkButton(
            email_actions,
            text="Delete",
            width=100,
            height=30,
            fg_color="transparent",
            text_color=COLORS["red"],
            hover_color="#FEE2E2",
            command=delete_teacher_email,
        ).pack(side="left", padx=(8, 0))
        definitions = [
            ("host", "SMTP Host", settings["smtp_host"], False),
            ("port", "SMTP Port", str(settings["smtp_port"]), False),
            ("sender", "Sender Email", settings.get("sender_email", ""), False),
            ("password", "App Password", "", True),
            ("subject", "Subject", "Attendance Report", False),
        ]
        for key, label, default, secret in definitions:
            ctk.CTkLabel(dialog, text=label).pack(anchor="w", padx=30, pady=(13, 2))
            fields[key] = ctk.CTkEntry(dialog, show="*" if secret else "")
            fields[key].insert(0, default)
            fields[key].pack(fill="x", padx=30)
        attachment = ctk.CTkFrame(dialog, corner_radius=10, fg_color="#EAF2FB")
        attachment.pack(fill="x", padx=30, pady=(18, 0))
        ctk.CTkLabel(
            attachment,
            text="📎  attendance_report.csv",
            font=("Segoe UI", 14, "bold"),
            text_color=COLORS["navy"],
        ).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(
            attachment,
            text="Automatically generated and attached when you send this report.",
            font=("Segoe UI", 11),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=16, pady=(0, 12))
        status = ctk.CTkLabel(
            dialog, text="", text_color=COLORS["muted"], font=("Segoe UI", 11)
        )
        status.pack(anchor="w", padx=30, pady=(10, 0))
        actions = ctk.CTkFrame(dialog, fg_color="transparent")
        actions.pack(fill="x", padx=30, pady=24)
        ctk.CTkButton(
            actions,
            text="Cancel",
            width=130,
            fg_color="transparent",
            text_color=COLORS["muted"],
            command=dialog.destroy,
        ).pack(side="right")
        send_button = ctk.CTkButton(
            actions,
            text="Send Report",
            width=210,
            fg_color=COLORS["green"],
            command=lambda: self._send_email(dialog, fields, send_button, status),
        )
        send_button.pack(side="right", padx=(0, 10))

    def _send_email(self, dialog, fields, send_button, status) -> None:
        data = {key: item.get().strip() for key, item in fields.items()}
        if not all(data.values()):
            messagebox.showerror(
                "Missing information",
                "Complete all fields before sending.",
                parent=dialog,
            )
            return
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", data["recipient"]):
            messagebox.showerror(
                "Invalid email", "Enter a valid teacher email address.", parent=dialog
            )
            return
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", data["sender"]):
            messagebox.showerror(
                "Invalid email", "Enter a valid sender email address.", parent=dialog
            )
            return
        try:
            port = int(data["port"])
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid SMTP port",
                "Enter an SMTP port between 1 and 65535.",
                parent=dialog,
            )
            return
        saved_recipients = get_settings().get("teacher_emails", [])
        if not isinstance(saved_recipients, list):
            saved_recipients = []
        recipients = list(dict.fromkeys([data["recipient"], *saved_recipients]))
        save_settings(
            {
                "teacher_email": data["recipient"],
                "teacher_emails": recipients,
                "smtp_host": data["host"],
                "smtp_port": port,
                "sender_email": data["sender"],
            }
        )
        app_password = data.pop("password")
        fields["password"].delete(0, "end")
        send_button.configure(state="disabled")
        status.configure(text="Sending...")
        self.notify("Sending email report…", "warning")

        def worker(password: str):
            try:
                report = generate_attendance_csv()
                email = build_email(
                    data["sender"], data["recipient"], data["subject"], report
                )
                send_email(
                    smtp_host=data["host"],
                    smtp_port=port,
                    sender=data["sender"],
                    app_password=password,
                    message=email,
                )
                self.after(0, lambda: self._email_succeeded(dialog))
            except Exception as error:
                self.after(
                    0,
                    lambda caught_error=error: self._email_failed(
                        dialog, send_button, status, caught_error
                    ),
                )
            finally:
                password = ""

        threading.Thread(target=worker, args=(app_password,), daemon=True).start()
        app_password = ""

    def _email_succeeded(self, dialog) -> None:
        if dialog.winfo_exists():
            dialog.destroy()
        self.notify("Attendance report has been emailed successfully.")
        messagebox.showinfo(
            "Success", "Attendance report has been emailed successfully.", parent=self
        )

    def _email_failed(self, dialog, send_button, status, error: Exception) -> None:
        if isinstance(error, smtplib.SMTPAuthenticationError):
            detail = "Authentication failed. Use the sender's Google App Password, not its Gmail password."
        elif isinstance(error, smtplib.SMTPConnectError):
            detail = (
                "Could not connect to the SMTP server. Check the SMTP host and port."
            )
        elif isinstance(error, (TimeoutError, socket.timeout)):
            detail = "The connection timed out. Check your internet connection and try again."
        elif isinstance(error, FileNotFoundError):
            detail = "The attendance CSV could not be generated. Please try again."
        elif isinstance(error, (socket.gaierror, OSError)):
            detail = "A network error occurred. Check your internet connection and SMTP settings."
        else:
            detail = "The report could not be emailed. Verify the SMTP settings and try again."
        if dialog.winfo_exists():
            send_button.configure(state="normal")
            status.configure(text="")
        self.notify(detail, "error")
        messagebox.showerror("Email report failed", detail, parent=self)


def main() -> None:
    configure_logging()
    ensure_runtime_directories()
    database.init_db()
    AttendanceApp().mainloop()


if __name__ == "__main__":
    main()
