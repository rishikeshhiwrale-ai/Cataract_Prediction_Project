
import pickle  # noqa: I001
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PDFImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from skimage.feature import hog


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "saved_model" / "cataract_svm.pkl"
IMAGE_SIZE = (224, 224)
APP_BG = "#f3f6fb"
NAVY = "#17324d"
BLUE = "#2463a6"
GREEN = "#18794e"
MUTED = "#64748b"


def image_features(image_path: Path) -> np.ndarray:
    """Extract HOG features; keep settings aligned with train_model.py."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The selected file could not be opened as an image.")
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
    ).reshape(1, -1)


class CataractApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VisionCheck | Cataract Screening Demo")
        self.root.geometry("900x760")
        self.root.minsize(820, 700)
        self.root.configure(bg=APP_BG)
        self.selected_image: Path | None = None
        self.prediction_text: str | None = None
        self.preview_ref = None
        self._style()
        self._build()

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", padding=7, font=("Segoe UI", 10))
        style.configure("Section.TLabelframe", background="white", padding=16)
        style.configure("Section.TLabelframe.Label", background="white", foreground=NAVY,
                        font=("Segoe UI", 11, "bold"))

    def _build(self):
        header = tk.Frame(self.root, bg=NAVY, padx=30, pady=22)
        header.pack(fill="x")
        tk.Label(header, text="VisionCheck", bg=NAVY, fg="white",
                 font=("Segoe UI", 23, "bold")).pack(anchor="w")
        tk.Label(header, text="Cataract image classification • Educational screening prototype",
                 bg=NAVY, fg="#d9e7f5", font=("Segoe UI", 10)).pack(anchor="w", pady=(3, 0))

        body = tk.Frame(self.root, bg=APP_BG, padx=26, pady=20)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        patient = ttk.LabelFrame(body, text="  Patient details  ", style="Section.TLabelframe")
        patient.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        patient.columnconfigure(1, weight=1)
        self.first_name = self._entry(patient, "First name", 0)
        self.last_name = self._entry(patient, "Last name", 1)
        self.age = self._combo(patient, "Age", [str(i) for i in range(1, 121)], 2)
        self.gender = self._combo(patient, "Gender", ["Female", "Male", "Other", "Prefer not to say"], 3)
        self.mobile = self._entry(patient, "Mobile number (optional)", 4)
        self.symptoms = self._combo(patient, "Reported vision concern", [
            "Not recorded", "Blurred or cloudy vision", "Glare or halos",
            "Reduced night vision", "Colors seem faded", "Other / multiple",
        ], 5)
        self.notes = tk.Text(patient, height=4, wrap="word", font=("Segoe UI", 10),
                             relief="solid", bd=1, highlightthickness=0)
        self._field_label(patient, "Additional notes (optional)", 6)
        self.notes.grid(row=6, column=1, sticky="ew", padx=(12, 0), pady=7)
        tk.Label(patient, text="Only include information needed for this demonstration.",
                 bg="white", fg=MUTED, font=("Segoe UI", 9)).grid(row=7, column=1, sticky="w", padx=12)

        image_card = ttk.LabelFrame(body, text="  Eye image  ", style="Section.TLabelframe")
        image_card.grid(row=0, column=1, sticky="nsew")
        image_card.columnconfigure(0, weight=1)
        image_card.rowconfigure(0, weight=1)
        self.preview = tk.Label(image_card, text="Choose an image to preview it here",
                                bg="#f1f5f9", fg=MUTED, width=30, height=12,
                                font=("Segoe UI", 10), wraplength=220)
        self.preview.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.image_label = tk.Label(image_card, text="No image selected", bg="white", fg=MUTED,
                                    font=("Segoe UI", 9), wraplength=230)
        self.image_label.grid(row=1, column=0, sticky="w", pady=4)
        self._button(image_card, "Choose eye image", self.choose_image, BLUE).grid(row=2, column=0, sticky="ew", pady=(8, 5))
        self._button(image_card, "Run prediction", self.predict, GREEN).grid(row=3, column=0, sticky="ew", pady=5)
        self.result_label = tk.Label(image_card, text="Prediction will appear here", bg="white",
                                     fg=NAVY, font=("Segoe UI", 11, "bold"), wraplength=230)
        self.result_label.grid(row=4, column=0, sticky="ew", pady=8)
        self.report_button = self._button(image_card, "Create detailed PDF report", self.create_report, "#d99019")
        self.report_button.grid(row=5, column=0, sticky="ew", pady=5)
        self.report_button.configure(state="disabled")

        footer = tk.Label(self.root,
            text="DEMONSTRATION ONLY  •  This model output is not a diagnosis or a substitute for an eye examination.",
            bg="#e7edf5", fg="#45566b", font=("Segoe UI", 9), padx=12, pady=11)
        footer.pack(fill="x", side="bottom")

    def _field_label(self, parent, label, row):
        tk.Label(parent, text=label, bg="white", fg="#334155", font=("Segoe UI", 10)).grid(
            row=row, column=0, sticky="w", pady=7)

    def _entry(self, parent, label, row):
        self._field_label(parent, label, row)
        entry = ttk.Entry(parent, font=("Segoe UI", 10))
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=7, ipady=5)
        return entry

    def _combo(self, parent, label, values, row):
        self._field_label(parent, label, row)
        box = ttk.Combobox(parent, values=values, state="readonly", font=("Segoe UI", 10))
        box.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=7)
        box.set("")
        return box

    def _button(self, parent, label, command, color):
        return tk.Button(parent, text=label, command=command, bg=color, fg="white",
                         activebackground=color, activeforeground="white", relief="flat",
                         cursor="hand2", padx=12, pady=10, font=("Segoe UI", 10, "bold"))

    def choose_image(self):
        chosen = filedialog.askopenfilename(
            title="Choose an eye image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not chosen:
            return
        self.selected_image = Path(chosen)
        self.prediction_text = None
        self.report_button.configure(state="disabled")
        self.result_label.configure(text="Run a prediction for this image", fg=NAVY)
        self.image_label.configure(text=self.selected_image.name, fg=GREEN)
        try:
            image = Image.open(self.selected_image).convert("RGB")
            image.thumbnail((270, 240))
            self.preview_ref = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.preview_ref, text="", width=270, height=240)
        except (OSError, ValueError) as exc:
            self.preview.configure(image="", text=f"Preview unavailable\n{exc}")

    def _patient_data(self):
        first, last = self.first_name.get().strip(), self.last_name.get().strip()
        if not first or not last:
            raise ValueError("Enter the patient's first and last name.")
        age = self.age.get().strip()
        if not age:
            raise ValueError("Choose an age.")
        mobile = self.mobile.get().strip()
        if mobile and not re.fullmatch(r"[+()\d\s.-]{7,20}", mobile):
            raise ValueError("Check the mobile number format, or leave it blank.")
        return {
            "Patient name": f"{first} {last}",
            "Age": age,
            "Gender": self.gender.get().strip() or "Not recorded",
            "Mobile": mobile or "Not provided",
            "Reported vision concern": self.symptoms.get().strip() or "Not recorded",
            "Additional notes": self.notes.get("1.0", "end").strip() or "None provided",
        }

    def predict(self):
        if not self.selected_image or not self.selected_image.exists():
            messagebox.showerror("Image required", "Choose an eye image first.")
            return
        if not MODEL_PATH.exists():
            messagebox.showerror("Model not found",
                f"The trained model is missing:\n{MODEL_PATH}\n\nPlace cataract_svm.pkl in saved_model.")
            return
        try:
            with MODEL_PATH.open("rb") as file:
                saved = pickle.load(file)
            # New training files save metadata alongside the estimator. Keep
            # compatibility with older files that contain just an SVC.
            model = saved.get("model") if isinstance(saved, dict) else saved
            if model is None or not hasattr(model, "predict"):
                raise ValueError("The saved model file does not contain a usable classifier.")
            result = int(model.predict(image_features(self.selected_image))[0])
            self.prediction_text = "Cataract class predicted" if result == 1 else "Normal class predicted"
            self.result_label.configure(text=self.prediction_text,
                                        fg="#b42318" if result == 1 else GREEN)
            self.report_button.configure(state="normal")
        except (OSError, pickle.UnpicklingError, EOFError, ValueError,
                AttributeError, TypeError, cv2.error) as exc:
            messagebox.showerror("Prediction failed", str(exc))

    def create_report(self):
        if not self.selected_image or not self.selected_image.exists() or not self.prediction_text:
            messagebox.showerror("Prediction required", "Choose an image and run prediction first.")
            return
        try:
            patient = self._patient_data()
        except ValueError as exc:
            messagebox.showerror("Check patient details", str(exc))
            return
        destination = filedialog.asksaveasfilename(
            title="Save screening report", defaultextension=".pdf",
            initialfile="visioncheck_report.pdf", filetypes=[("PDF files", "*.pdf")])
        if not destination:
            return
        try:
            self._write_pdf(Path(destination), patient)
            messagebox.showinfo("Report created", f"PDF saved successfully:\n{destination}")
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Could not create report", str(exc))

    def _write_pdf(self, destination: Path, patient: dict):
        created = datetime.now().astimezone().strftime("%d %B %Y, %I:%M %p %Z")
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                                  fontSize=19, textColor=colors.HexColor(NAVY), alignment=TA_LEFT,
                                  spaceAfter=4))
        styles.add(ParagraphStyle(name="SmallMuted", parent=styles["Normal"], fontSize=8,
                                  textColor=colors.HexColor(MUTED), leading=11))
        styles.add(ParagraphStyle(name="SectionHead", parent=styles["Heading2"], fontSize=12,
                                  textColor=colors.HexColor(BLUE), spaceBefore=12, spaceAfter=7))
        doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                                topMargin=17*mm, bottomMargin=18*mm,
                                title="VisionCheck Cataract Screening Report", author="VisionCheck")
        story = [Paragraph("VisionCheck", styles["ReportTitle"]),
                 Paragraph("CATARACT IMAGE CLASSIFICATION · EDUCATIONAL DEMONSTRATION", styles["SmallMuted"]),
                 Spacer(1, 8)]
        banner = Table([[Paragraph(f"<b>Model output:</b> {self.prediction_text}", styles["Normal"]),
                         Paragraph(f"<b>Report date:</b> {created}", styles["Normal"])]], colWidths=[83*mm, 83*mm])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#eaf1f8")),
            ("BOX", (0,0), (-1,-1), .5, colors.HexColor("#cbd8e6")),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 9),
            ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ]))
        story += [banner, Paragraph("Patient information", styles["SectionHead"])]
        rows = [[Paragraph(f"<b>{key}</b>", styles["Normal"]),
                 Paragraph(self._escape(value), styles["Normal"])] for key, value in patient.items()]
        info = Table(rows, colWidths=[48*mm, 118*mm], hAlign="LEFT")
        info.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f1f5f9")),
            ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d7e0ea")),
            ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ]))
        story += [info, Paragraph("Image used for this prediction", styles["SectionHead"]),
                  Paragraph(f"File: {self._escape(self.selected_image.name)}", styles["Normal"]), Spacer(1, 7)]
        try:
            with Image.open(self.selected_image) as im:
                im.verify()
            picture = PDFImage(str(self.selected_image), width=110*mm, height=75*mm, kind="proportional")
            story.append(picture)
        except (OSError, ValueError):
            story.append(Paragraph("Image preview could not be embedded in this PDF.", styles["SmallMuted"]))
        story += [Paragraph("How to read this result", styles["SectionHead"]),
                  Paragraph("This report records the output of a machine-learning classifier applied to the selected image. "
                            "It does not establish whether cataract is present and does not assess overall eye health. "
                            "Image quality, camera differences, training data, and model limitations can affect the output.",
                            styles["Normal"]),
                  Paragraph("Recommended next step", styles["SectionHead"]),
                  Paragraph("Discuss vision concerns and this experimental result with a qualified eye-care professional. "
                            "Do not delay an examination or treatment based on this report.", styles["Normal"]),
                  Spacer(1, 12),
                  KeepTogether([Paragraph("Educational prototype — not a medical diagnosis.",
                                           ParagraphStyle("notice", parent=styles["Normal"], fontName="Helvetica-Bold",
                                                          textColor=colors.HexColor("#9b2c2c"), alignment=TA_CENTER)),
                                Paragraph("Generated by VisionCheck. This document contains personal information; "
                                          "store and share it carefully.",
                                          ParagraphStyle("privacy", parent=styles["SmallMuted"], alignment=TA_CENTER))])]
        doc.build(story)

    @staticmethod
    def _escape(text: str) -> str:
        return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br/>"))


if __name__ == "__main__":
    app_root = tk.Tk()
    CataractApp(app_root)
    app_root.mainloop()
