import sys
import json
import re
import numpy as np
import requests
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QListWidget, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy, QSplitter,
    QTabWidget, QCheckBox, QLineEdit, QPushButton, QScrollArea,
    QTextEdit, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, QRect, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import QtInteractor

AFFECTED_COLOR = "#ff2d55"

STAGE_COLORS = {
    "Early":                    "#a6e3a1",
    "Infection":                "#a6e3a1",
    "Subclinical":              "#a6e3a1",
    "Asymptomatic":             "#a6e3a1",
    "Developing":               "#f9e2af",
    "Febrile":                  "#f9e2af",
    "Remission":                "#f9e2af",
    "Early Inflammatory":       "#f9e2af",
    "Acute":                    "#fab387",
    "Elevated":                 "#fab387",
    "Recurrent":                "#fab387",
    "Severe":                   "#f38ba8",
    "Intoxication":             "#f38ba8",
    "Neuroinvasive":            "#f38ba8",
    "Chronic / Elephantiasis":  "#cba6f7",
    "Critical":                 "#ff2d55",
    "Severe (Rare)":            "#ff2d55",
}

# Organ risk weights per disease (0-10 scale)
ORGAN_RISK = {
    "Malaria":              {"brain":9,"liver":8,"spleen":9,"kidney":7,"blood":6},
    "Dengue":               {"liver":7,"blood":9,"skin":6,"muscles":7,"joints":5},
    "Chikungunya":          {"joints":9,"muscles":8,"skin":6,"lymph_nodes":5},
    "Zika":                 {"brain":8,"eyes":7,"skin":5,"joints":4},
    "Yellow Fever":         {"liver":10,"kidney":8,"heart":7,"skin":6,"blood":5},
    "West Nile Fever":      {"brain":8,"spinal_cord":7,"lymph_nodes":5,"skin":4},
    "Lymphatic Filariasis": {"lymph_nodes":9,"limbs":10,"skin":7},
}

# ── Temperature conversion helpers ───────────────────────────────────────────
def c_to_f(c): return round(c * 9/5 + 32, 1)
def f_to_c(f): return (f - 32) * 5/9

# Typical fever curve per disease: list of (day, temp_in_FAHRENHEIT)
FEVER_CURVES = {
    "Malaria":              [(0,c_to_f(37.0)),(1,c_to_f(38.5)),(2,c_to_f(40.5)),(3,c_to_f(39.0)),(4,c_to_f(40.8)),(5,c_to_f(39.2)),(6,c_to_f(41.0)),(7,c_to_f(38.0)),(8,c_to_f(37.2))],
    "Dengue":               [(0,c_to_f(37.0)),(1,c_to_f(39.5)),(2,c_to_f(40.2)),(3,c_to_f(40.0)),(4,c_to_f(39.8)),(5,c_to_f(38.5)),(6,c_to_f(37.8)),(7,c_to_f(37.2))],
    "Chikungunya":          [(0,c_to_f(37.0)),(1,c_to_f(39.5)),(2,c_to_f(39.8)),(3,c_to_f(38.5)),(4,c_to_f(37.8)),(5,c_to_f(37.5)),(6,c_to_f(37.2))],
    "Zika":                 [(0,c_to_f(37.0)),(1,c_to_f(38.2)),(2,c_to_f(38.5)),(3,c_to_f(38.2)),(4,c_to_f(37.8)),(5,c_to_f(37.4)),(6,c_to_f(37.1))],
    "Yellow Fever":         [(0,c_to_f(37.0)),(1,c_to_f(39.5)),(2,c_to_f(40.2)),(3,c_to_f(38.5)),(4,c_to_f(37.5)),(5,c_to_f(40.0)),(6,c_to_f(40.8)),(7,c_to_f(39.0))],
    "West Nile Fever":      [(0,c_to_f(37.0)),(1,c_to_f(38.5)),(2,c_to_f(39.2)),(3,c_to_f(39.0)),(4,c_to_f(38.5)),(5,c_to_f(38.0)),(6,c_to_f(37.5))],
    "Lymphatic Filariasis": [(0,c_to_f(37.0)),(1,c_to_f(38.5)),(2,c_to_f(38.8)),(3,c_to_f(37.8)),(4,c_to_f(38.2)),(5,c_to_f(38.0)),(6,c_to_f(37.5))],
}


# ── Symptom checker: all known symptoms mapped to diseases ────────────────────
ALL_SYMPTOMS = [
    "Fever",
    "Chills / Rigors",
    "Headache",
    "Muscle pain (Myalgia)",
    "Joint pain (Arthralgia)",
    "Joint swelling",
    "Fatigue / Weakness",
    "Nausea / Vomiting",
    "Skin rash",
    "Pain behind eyes (Retro-orbital)",
    "Bleeding gums / Bruising",
    "Swollen lymph nodes",
    "Jaundice (Yellow skin/eyes)",
    "Red eyes (Conjunctivitis)",
    "Neck stiffness",
    "Confusion / Disorientation",
    "Seizures",
    "Limb swelling",
    "Skin thickening",
    "Enlarged spleen",
    "Abdominal pain",
    "Dark urine",
    "Anaemia",
    "Cough / Breathing difficulty",
]

# Each disease → list of symptom keywords that match ALL_SYMPTOMS entries
DISEASE_SYMPTOMS = {
    "Malaria":              ["Fever","Chills / Rigors","Headache","Muscle pain (Myalgia)",
                             "Fatigue / Weakness","Nausea / Vomiting","Enlarged spleen",
                             "Anaemia","Jaundice (Yellow skin/eyes)","Confusion / Disorientation",
                             "Seizures","Abdominal pain"],
    "Dengue":               ["Fever","Headache","Muscle pain (Myalgia)","Joint pain (Arthralgia)",
                             "Fatigue / Weakness","Nausea / Vomiting","Skin rash",
                             "Pain behind eyes (Retro-orbital)","Bleeding gums / Bruising",
                             "Abdominal pain","Dark urine"],
    "Chikungunya":          ["Fever","Joint pain (Arthralgia)","Joint swelling","Muscle pain (Myalgia)",
                             "Skin rash","Headache","Fatigue / Weakness","Swollen lymph nodes"],
    "Zika":                 ["Fever","Skin rash","Red eyes (Conjunctivitis)","Joint pain (Arthralgia)",
                             "Muscle pain (Myalgia)","Headache","Fatigue / Weakness"],
    "Yellow Fever":         ["Fever","Chills / Rigors","Headache","Muscle pain (Myalgia)",
                             "Nausea / Vomiting","Jaundice (Yellow skin/eyes)","Abdominal pain",
                             "Dark urine","Bleeding gums / Bruising","Fatigue / Weakness"],
    "West Nile Fever":      ["Fever","Headache","Muscle pain (Myalgia)","Fatigue / Weakness",
                             "Skin rash","Swollen lymph nodes","Neck stiffness",
                             "Confusion / Disorientation","Seizures"],
    "Lymphatic Filariasis": ["Fever","Chills / Rigors","Limb swelling","Swollen lymph nodes",
                             "Skin thickening","Fatigue / Weakness","Cough / Breathing difficulty"],
}


class CameraVitalsWorker(QThread):
    data_signal = pyqtSignal(object, bool, bool) # vitals_data, red_eye_detected, rash_detected

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = True

    def run(self):
        # Resolve or download cascade XML files in background if missing
        import os
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        
        if not os.path.exists(face_path) or os.path.getsize(face_path) < 1000:
            face_path = "haarcascade_frontalface_default.xml"
            if not os.path.exists(face_path):
                print("Worker: Downloading face cascade XML...")
                try:
                    r = requests.get("https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml", timeout=10)
                    with open(face_path, "wb") as f:
                        f.write(r.content)
                except Exception as e:
                    print("Worker: Failed to download face cascade:", e)
                    
        if not os.path.exists(eye_path) or os.path.getsize(eye_path) < 1000:
            eye_path = "haarcascade_eye.xml"
            if not os.path.exists(eye_path):
                print("Worker: Downloading eye cascade XML...")
                try:
                    r = requests.get("https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_eye.xml", timeout=10)
                    with open(eye_path, "wb") as f:
                        f.write(r.content)
                except Exception as e:
                    print("Worker: Failed to download eye cascade:", e)

        face_cascade = cv2.CascadeClassifier(face_path)
        eye_cascade = cv2.CascadeClassifier(eye_path)

        # Open camera in background to avoid freezing GUI startup
        self.cap = cv2.VideoCapture(0)
        
        while self.running:
            # 1. Fetch vitals from the Flask simulator
            vitals_data = None
            try:
                r = requests.get("http://127.0.0.1:5000/data", timeout=2)
                if r.status_code == 200:
                    vitals_data = r.json()
            except Exception as e:
                print("Worker Vitals Fetch Error:", e)

            # 2. Analyze camera frame for symptoms
            red_eye_detected = False
            rash_detected = False
            
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Downsample frame by 50% for significantly faster cascade detection
                        h, w = frame.shape[:2]
                        scale = 0.5
                        small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                        
                        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        for (x, y, w_face, h_face) in faces:
                            roi_gray = gray[y:y+h_face, x:x+w_face]
                            roi_color = small_frame[y:y+h_face, x:x+w_face]
                            
                            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=6)
                            
                            for (ex, ey, ew, eh) in eyes:
                                if ey > h_face * 0.15 and ey + eh < h_face * 0.60:
                                    eye_region = roi_color[ey:ey+eh, ex:ex+ew]
                                    b, g, r_ch = cv2.split(eye_region)
                                    r_mean = np.mean(r_ch)
                                    g_mean = np.mean(g)
                                    b_mean = np.mean(b)
                                    
                                    if r_mean > 1.55 * g_mean and r_mean > 1.55 * b_mean and r_mean > 100:
                                        red_eye_detected = True
                        
                        # HSV skin rash detection on the downsampled frame
                        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                        lower_red1 = np.array([0, 130, 90])
                        upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([170, 130, 90])
                        upper_red2 = np.array([180, 255, 255])
                        
                        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                        mask = mask1 + mask2
                        
                        red_pixels = cv2.countNonZero(mask)
                        # Scale the original 80000 pixel threshold down to 20000 due to 0.5 resize factor
                        if red_pixels > 20000:
                            rash_detected = True
            except Exception as e:
                print("Worker Camera Analysis Error:", e)

            # Emit signal to update UI
            self.data_signal.emit(vitals_data, red_eye_detected, rash_detected)
            
            # Run every 2 seconds
            self.msleep(2000)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class DigitalTwinApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Twin - Human Body")
        self.setMinimumSize(1280, 820)
        self._current_disease = None

        self.red_eye_detected = False
        self.rash_detected = False
        
        self.current_tree = None
        self.current_node = None
        self.final_diagnosis = None

        with open("diseases.json", "r", encoding="utf-8") as f:
            self.diseases = json.load(f)

        self.live_symptoms_responses = {}
        self.last_live_vitals = None
        self.current_differentiating_symptom = None

        self.init_ui()
        
        self.yes_btn.clicked.connect(self.on_live_yes_clicked)
        self.no_btn.clicked.connect(self.on_live_no_clicked)
        
        QTimer.singleShot(300, self.load_body_model)

        # Start background worker thread for vitals and camera analysis
        self.worker = CameraVitalsWorker()
        self.worker.data_signal.connect(self.handle_worker_data)
        self.worker.start()

    # ─────────────────────────────────────────────────────────────────────────
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── LEFT PANEL with tabs ──────────────────────────────────────────────
        left_panel = QFrame()
        left_panel.setMinimumWidth(450)
        left_panel.setMaximumWidth(550)
        left_panel.setStyleSheet("QFrame{background:#1e1e2e;border-radius:10px;}")
        left_outer = QVBoxLayout(left_panel)
        left_outer.setContentsMargins(10, 12, 10, 12)
        left_outer.setSpacing(8)

        # Title
        t = QLabel("Digital Twin")
        t.setFont(QFont("Arial", 17, QFont.Weight.Bold))
        t.setStyleSheet("color:#cdd6f4;background:transparent;")
        left_outer.addWidget(t)

        s = QLabel("Mosquito-borne Disease Visualizer")
        s.setFont(QFont("Arial", 9))
        s.setStyleSheet("color:#6c7086;background:transparent;")
        s.setWordWrap(True)
        left_outer.addWidget(s)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#313244;max-height:1px;")
        left_outer.addWidget(sep)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane{background:#1e1e2e;border:none;}
            QTabBar::tab{background:#313244;color:#6c7086;padding:6px 10px;
                         font-size:11px;border-radius:4px;margin-right:3px;}
            QTabBar::tab:selected{background:#89b4fa;color:#1e1e2e;font-weight:bold;}
            QTabBar::tab:hover{background:#45475a;color:#cdd6f4;}
        """)
        left_outer.addWidget(self.tabs)

        # ── TAB 1: Disease Info ───────────────────────────────────────────────
        tab1 = QWidget()
        tab1.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(tab1)
        ll.setContentsMargins(0, 8, 0, 0)
        ll.setSpacing(8)
        self.tabs.addTab(tab1, "🦟  Disease")

        ll.addWidget(self._lbl("Select Disease", bold=True))
        self.combo = QComboBox()
        self.combo.addItem("-- Select a Disease --")
        for d in self.diseases:
            self.combo.addItem(d)
        self.combo.setStyleSheet("""
            QComboBox{background:#313244;color:#cdd6f4;border-radius:6px;
                      padding:6px;font-size:13px;border:none;}
            QComboBox QAbstractItemView{background:#313244;color:#cdd6f4;}""")
        self.combo.currentTextChanged.connect(self.on_disease_selected)
        ll.addWidget(self.combo)

        ll.addWidget(self._lbl("Mosquito Type", small=True))
        self.mosq_value = QLabel("—")
        self.mosq_value.setStyleSheet("color:#89b4fa;font-size:11px;background:transparent;")
        self.mosq_value.setWordWrap(True)
        ll.addWidget(self.mosq_value)

        ll.addWidget(self._lbl("Severity", small=True))
        self.severity_value = QLabel("—")
        self.severity_value.setStyleSheet("color:#cdd6f4;font-size:11px;background:transparent;")
        ll.addWidget(self.severity_value)

        ll.addWidget(self._lbl("Symptoms", bold=True))
        self.symptom_list = QListWidget()
        self.symptom_list.setStyleSheet("""
            QListWidget{background:#313244;color:#cdd6f4;border-radius:6px;
                        font-size:11px;padding:4px;border:none;}
            QListWidget::item{padding:4px;}
            QListWidget::item:hover{background:#45475a;border-radius:4px;}""")
        ll.addWidget(self.symptom_list)

        ll.addWidget(self._lbl("Affected Organs", bold=True, color="#ff2d55"))
        self.affected_list = QListWidget()
        self.affected_list.setMaximumHeight(100)
        self.affected_list.setStyleSheet("""
            QListWidget{background:#2a0a14;color:#ff2d55;border-radius:6px;
                        font-size:11px;padding:4px;border:1px solid #ff2d55;}
            QListWidget::item{padding:3px;}""")
        ll.addWidget(self.affected_list)

        ll.addWidget(self._lbl("Body Temperature", bold=True))
        self.temp_container = QWidget()
        self.temp_container.setFixedHeight(105)
        self.temp_container.setStyleSheet("background:transparent;")
        self.temp_layout = QVBoxLayout(self.temp_container)
        self.temp_layout.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.temp_container)
        self.draw_thermometer(37.0)

        # ── TAB 2: Symptom Checker ────────────────────────────────────────────
        tab2 = QWidget()

        checker_scroll = QScrollArea()
        checker_scroll.setWidgetResizable(True)
        checker_scroll.setFrameShape(QFrame.Shape.NoFrame)

        checker_content = QWidget()
        sl = QVBoxLayout(checker_content)

        checker_scroll.setWidget(checker_content)

        self.tabs.addTab(checker_scroll, "🔍  Checker")
        tab2.setStyleSheet("background:transparent;")
        
        sl.setContentsMargins(0, 8, 0, 0)
        sl.setSpacing(6)

        # ── STEP 1: Temperature input ─────────────────────────────────────────
        step1_frame = QFrame()
        step1_frame.setStyleSheet("""
            QFrame{background:#1e2a1e;border:1px solid #a6e3a1;border-radius:8px;}""")
        step1_v = QVBoxLayout(step1_frame)
        step1_v.setContentsMargins(10, 8, 10, 8)
        step1_v.setSpacing(5)

        step1_title = QLabel("Step 1 — Enter Body Temperature (°F)")
        step1_title.setStyleSheet(
            "color:#a6e3a1;font-size:10px;font-weight:bold;background:transparent;")
        step1_v.addWidget(step1_title)

        temp_row = QHBoxLayout()
        self.temp_input = QLineEdit()
        self.temp_input.setFixedHeight(40)
        self.temp_input.setMinimumWidth(180)
        self.temp_input.setPlaceholderText("e.g.  103.1")
        self.temp_input.setMaxLength(5)
        self.temp_input.setStyleSheet("""
            QLineEdit{background:#313244;color:#a6e3a1;border-radius:6px;
                      padding:7px;font-size:16px;font-weight:bold;
                      border:none;letter-spacing:1px;}
            QLineEdit:focus{border:1px solid #a6e3a1;}""")

        deg_lbl = QLabel("°F")
        deg_lbl.setStyleSheet(
            "color:#6c7086;font-size:14px;background:transparent;padding-left:4px;")

        temp_row.addWidget(self.temp_input, stretch=1)
        temp_row.addWidget(deg_lbl)
        step1_v.addLayout(temp_row)

        # Quick-pick temperature buttons
        quick_row = QHBoxLayout()
        quick_row.setSpacing(4)
        for val in ["99.5", "101.3", "103.1", "104.9", "106.7"]:
            btn = QPushButton(val)
            btn.setFixedSize(70, 24)
            btn.setStyleSheet("""
                QPushButton{background:#313244;color:#6c7086;border-radius:4px;
                            font-size:9px;border:none;padding:2px;}
                QPushButton:hover{background:#45475a;color:#cdd6f4;}""")
            btn.clicked.connect(lambda _, v=val: self.temp_input.setText(v))
            quick_row.addWidget(btn)
        step1_v.addLayout(quick_row)

        quick_hint = QLabel("Quick pick ↑  or type your temperature")
        quick_hint.setStyleSheet(
            "color:#45475a;font-size:8px;background:transparent;")
        step1_v.addWidget(quick_hint)
        sl.addWidget(step1_frame)

        # Temp-only analyse button
        self.temp_analyse_btn = QPushButton("🌡  Detect by Temperature")
        self.temp_analyse_btn.setStyleSheet("""
            QPushButton{background:#a6e3a1;color:#1e1e2e;border-radius:7px;
                        padding:7px;font-size:11px;font-weight:bold;border:none;}
            QPushButton:hover{background:#c3f0c3;}
            QPushButton:pressed{background:#7ec87e;}""")
        self.temp_analyse_btn.clicked.connect(self.run_temperature_check)
        sl.addWidget(self.temp_analyse_btn)

        # ── STEP 2: Optional symptoms ─────────────────────────────────────────
        step2_frame = QFrame()
        step2_frame.setStyleSheet("""
            QFrame{background:#1e1e2e;border:1px solid #45475a;border-radius:8px;}""")
        step2_v = QVBoxLayout(step2_frame)
        step2_v.setContentsMargins(10, 8, 10, 8)
        step2_v.setSpacing(5)

        step2_title = QLabel("Step 2 — Refine with Symptoms  (optional)")
        step2_title.setStyleSheet(
            "color:#89b4fa;font-size:10px;font-weight:bold;background:transparent;")
        step2_v.addWidget(step2_title)

        self.symptom_input = QLineEdit()
        self.symptom_input.setPlaceholderText("e.g. fever, joint pain, rash...")
        self.symptom_input.setStyleSheet("""
            QLineEdit{background:#313244;color:#cdd6f4;border-radius:6px;
                      padding:6px;font-size:11px;border:none;}
            QLineEdit:focus{border:1px solid #89b4fa;}""")
        step2_v.addWidget(self.symptom_input)
        sl.addWidget(step2_frame)

        sl.addWidget(self._lbl("Or tick symptoms", bold=True))

        # Scrollable checkbox area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea{background:#313244;border-radius:6px;}
            QScrollBar:vertical{background:#313244;width:5px;border-radius:2px;}
            QScrollBar::handle:vertical{background:#45475a;border-radius:2px;}""")

        chk_widget = QWidget()
        chk_widget.setStyleSheet("background:#313244;")
        chk_grid = QVBoxLayout(chk_widget)
        chk_grid.setContentsMargins(6, 6, 6, 6)
        chk_grid.setSpacing(3)

        self.symptom_checkboxes = {}
        for sym in ALL_SYMPTOMS:
            cb = QCheckBox(sym)
            cb.setStyleSheet("""
                QCheckBox{color:#cdd6f4;font-size:10px;background:transparent;}
                QCheckBox::indicator{width:13px;height:13px;border-radius:3px;
                    border:1px solid #45475a;background:#1e1e2e;}
                QCheckBox::indicator:checked{background:#89b4fa;border:1px solid #89b4fa;}
                QCheckBox:hover{color:#89b4fa;}""")
            self.symptom_checkboxes[sym] = cb
            chk_grid.addWidget(cb)

        scroll.setWidget(chk_widget)
        scroll.setFixedHeight(100)
        sl.addWidget(scroll)

        # ── STEP 3: Travel History / Geographic Context ───────────────────────────
        step3_frame = QFrame()
        step3_frame.setStyleSheet("""
            QFrame{background:#1e1e2e;border:1px solid #45475a;border-radius:8px;}""")
        step3_v = QVBoxLayout(step3_frame)
        step3_v.setContentsMargins(10, 8, 10, 8)
        step3_v.setSpacing(5)

        step3_title = QLabel("Step 3 — Travel History / Geographic Context")
        step3_title.setStyleSheet(
            "color:#cba6f7;font-size:10px;font-weight:bold;background:transparent;")
        step3_v.addWidget(step3_title)

        self.travel_input = QComboBox()
        self.travel_input.addItem("No Recent Travel / Local", "none")
        self.travel_input.addItem("Sub-Saharan Africa", "africa")
        self.travel_input.addItem("South / Central America", "americas")
        self.travel_input.addItem("South / Southeast Asia", "asia")
        self.travel_input.addItem("Other / Non-Endemic Regions", "other")
        self.travel_input.setStyleSheet("""
            QComboBox{background:#313244;color:#cdd6f4;border-radius:6px;
                      padding:6px;font-size:11px;border:none;}
            QComboBox QAbstractItemView{background:#313244;color:#cdd6f4;}""")
        step3_v.addWidget(self.travel_input)
        sl.addWidget(step3_frame)
        
        self.travel_input.currentIndexChanged.connect(lambda: self.fetch_vitals())

        # Combined analyse button
        self.analyse_btn = QPushButton("🔬  Analyse Temp + Symptoms")
        self.analyse_btn.setStyleSheet("""
            QPushButton{background:#89b4fa;color:#1e1e2e;border-radius:7px;
                        padding:7px;font-size:11px;font-weight:bold;border:none;}
            QPushButton:hover{background:#b4d0fa;}
            QPushButton:pressed{background:#6b9cde;}""")
        self.analyse_btn.clicked.connect(self.run_symptom_checker)
        sl.addWidget(self.analyse_btn)
        self.final_btn = QPushButton("🧠 Final Diagnosis")

        self.final_btn.setStyleSheet("""
            QPushButton{
            background:#a6e3a1;
            color:#11111b;
            border-radius:7px;
            padding:7px;
            font-size:11px;
            font-weight:bold;
            border:none;
            }
        """)

        self.final_btn.clicked.connect(
            self.start_final_diagnosis
        )

        sl.addWidget(self.final_btn)
        # Clear button
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("""
            QPushButton{background:#313244;color:#6c7086;border-radius:6px;
                        padding:5px;font-size:10px;border:none;}
            QPushButton:hover{background:#45475a;color:#cdd6f4;}""")
        clear_btn.clicked.connect(self.clear_checker)
        sl.addWidget(clear_btn)

        # Result area
        self.checker_result = QTextEdit()
        self.checker_result.setReadOnly(True)
        self.checker_result.setFixedHeight(220)
        self.checker_result.setStyleSheet("""
            QTextEdit{background:#11111b;color:#cdd6f4;border-radius:7px;
                      font-size:10px;padding:8px;border:1px solid #313244;}
            QScrollBar:vertical{background:#11111b;width:5px;border-radius:2px;}
            QScrollBar::handle:vertical{background:#45475a;border-radius:2px;}""")
        self.checker_result.setPlaceholderText(
            "Enter temperature, tick symptoms, and click Final Diagnosis.")
        sl.addWidget(self.checker_result)
        
        # Live Vitals Card
        vitals_frame = QFrame()
        vitals_frame.setStyleSheet("""
        QFrame{
            background:#11111b;
            border-radius:8px;
            padding:8px;
        }
        """)

        vitals_layout = QVBoxLayout(vitals_frame)

        title = QLabel("🩺 Enter Vitals")
        title.setStyleSheet("color:#89b4fa;font-weight:bold;font-size:12px;")
        
        self.live_mode_cb = QCheckBox("Live Simulator")
        self.live_mode_cb.setChecked(False)
        self.live_mode_cb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.live_mode_cb.setStyleSheet("""
            QCheckBox{color:#89b4fa;font-size:11px;font-weight:bold;background:transparent;}
            QCheckBox::indicator{width:14px;height:14px;border-radius:4px;
                border:1px solid #45475a;background:#1e1e2e;}
            QCheckBox::indicator:checked{background:#89b4fa;border:1px solid #89b4fa;}
        """)
        self.live_mode_cb.toggled.connect(self.on_live_mode_changed)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(title)
        mode_layout.addStretch()
        mode_layout.addWidget(self.live_mode_cb)
        vitals_layout.addLayout(mode_layout)

        # Style for the inputs
        input_style = """
            QLineEdit{
                background:#313244;
                color:#cdd6f4;
                border-radius:4px;
                padding:4px;
                font-size:11px;
                font-weight:bold;
                border:1px solid #45475a;
            }
            QLineEdit:focus{
                border:1px solid #89b4fa;
            }
            QLineEdit:disabled{
                background:#11111b;
                color:#6c7086;
                border:none;
            }
        """

        # Temperature
        self.temp_label = QLabel("🌡 Temp:")
        self.temp_label.setStyleSheet("color:#cdd6f4;font-size:11px;background:transparent;")
        self.temp_val_input = QLineEdit()
        self.temp_val_input.setStyleSheet(input_style)
        self.temp_val_input.setPlaceholderText("e.g. 101.3")
        self.temp_val_input.setFixedWidth(65)
        self.temp_val_input.setEnabled(True)
        self.temp_val_unit = QLabel("°F")
        self.temp_val_unit.setStyleSheet("color:#6c7086;font-size:11px;background:transparent;")

        # Heart Rate
        self.hr_label = QLabel("❤️ HR:")
        self.hr_label.setStyleSheet("color:#cdd6f4;font-size:11px;background:transparent;")
        self.hr_val_input = QLineEdit()
        self.hr_val_input.setStyleSheet(input_style)
        self.hr_val_input.setPlaceholderText("e.g. 80")
        self.hr_val_input.setFixedWidth(65)
        self.hr_val_input.setEnabled(True)
        self.hr_val_unit = QLabel("BPM")
        self.hr_val_unit.setStyleSheet("color:#6c7086;font-size:11px;background:transparent;")

        # SpO2
        self.spo2_label = QLabel("🫁 SpO₂:")
        self.spo2_label.setStyleSheet("color:#cdd6f4;font-size:11px;background:transparent;")
        self.spo2_val_input = QLineEdit()
        self.spo2_val_input.setStyleSheet(input_style)
        self.spo2_val_input.setPlaceholderText("e.g. 98")
        self.spo2_val_input.setFixedWidth(65)
        self.spo2_val_input.setEnabled(True)
        self.spo2_val_unit = QLabel("%")
        self.spo2_val_unit.setStyleSheet("color:#6c7086;font-size:11px;background:transparent;")

        # BP
        self.bp_label = QLabel("🩸 BP:")
        self.bp_label.setStyleSheet("color:#cdd6f4;font-size:11px;background:transparent;")
        self.bp_sys_input = QLineEdit()
        self.bp_sys_input.setStyleSheet(input_style)
        self.bp_sys_input.setPlaceholderText("sys")
        self.bp_sys_input.setFixedWidth(35)
        self.bp_sys_input.setEnabled(True)
        
        self.bp_slash = QLabel("/")
        self.bp_slash.setStyleSheet("color:#6c7086;font-size:11px;font-weight:bold;background:transparent;")
        
        self.bp_dia_input = QLineEdit()
        self.bp_dia_input.setStyleSheet(input_style)
        self.bp_dia_input.setPlaceholderText("dia")
        self.bp_dia_input.setFixedWidth(35)
        self.bp_dia_input.setEnabled(True)
        
        self.bp_val_unit = QLabel("mmHg")
        self.bp_val_unit.setStyleSheet("color:#6c7086;font-size:11px;background:transparent;")

        # Connect text changes to automatic diagnostic calculations when user edits vitals manually
        self.temp_val_input.textChanged.connect(self.run_manual_vitals_check)
        self.hr_val_input.textChanged.connect(self.run_manual_vitals_check)
        self.spo2_val_input.textChanged.connect(self.run_manual_vitals_check)
        self.bp_sys_input.textChanged.connect(self.run_manual_vitals_check)
        self.bp_dia_input.textChanged.connect(self.run_manual_vitals_check)

        form_layout = QGridLayout()
        form_layout.setSpacing(6)
        form_layout.setContentsMargins(0, 4, 0, 4)

        form_layout.addWidget(self.temp_label, 0, 0, Qt.AlignmentFlag.AlignRight)
        form_layout.addWidget(self.temp_val_input, 0, 1)
        form_layout.addWidget(self.temp_val_unit, 0, 2)
        
        form_layout.addWidget(self.hr_label, 1, 0, Qt.AlignmentFlag.AlignRight)
        form_layout.addWidget(self.hr_val_input, 1, 1)
        form_layout.addWidget(self.hr_val_unit, 1, 2)
        
        form_layout.addWidget(self.spo2_label, 2, 0, Qt.AlignmentFlag.AlignRight)
        form_layout.addWidget(self.spo2_val_input, 2, 1)
        form_layout.addWidget(self.spo2_val_unit, 2, 2)
        
        form_layout.addWidget(self.bp_label, 3, 0, Qt.AlignmentFlag.AlignRight)
        bp_input_layout = QHBoxLayout()
        bp_input_layout.setSpacing(2)
        bp_input_layout.setContentsMargins(0, 0, 0, 0)
        bp_input_layout.addWidget(self.bp_sys_input)
        bp_input_layout.addWidget(self.bp_slash)
        bp_input_layout.addWidget(self.bp_dia_input)
        form_layout.addLayout(bp_input_layout, 3, 1)
        form_layout.addWidget(self.bp_val_unit, 3, 2)

        vitals_layout.addLayout(form_layout)

        self.status_live = QLabel("🟢 Patient Status : Stable")
        self.status_live.setStyleSheet("""
        color:#a6e3a1;
        font-size:12px;
        font-weight:bold;
        """)
        vitals_layout.addWidget(self.status_live)
        self.possible_disease_lbl = QLabel(
            "🦟 Possible Diseases:\n\nWaiting for vitals..."
        )
        self.possible_disease_lbl.setWordWrap(True)
        self.possible_disease_lbl.setMinimumHeight(55)
        self.possible_disease_lbl.setStyleSheet("""
            color:#cdd6f4;
            font-size:11px;
            background:#1e1e2e;
            padding:8px;
            border-radius:6px;
        """)
        vitals_layout.addWidget(self.possible_disease_lbl)
        self.camera_status = QLabel(
            "📷 Camera Symptoms: None"
        )

        self.camera_status.setStyleSheet("""
            color:#f9e2af;
            background:#313244;
            padding:8px;
            border-radius:6px;
        """)

        vitals_layout.addWidget(self.camera_status)
        self.question_lbl = QLabel("Waiting for analysis...")
        self.question_lbl.setWordWrap(True)
        self.question_lbl.setMinimumHeight(35)
        self.question_lbl.setStyleSheet("""
            color:#f9e2af;
            background:#313244;
            padding:8px;
            border-radius:6px;
        """)

        vitals_layout.addWidget(self.question_lbl)
        self.question_lbl.hide()
        
        self.yes_btn = QPushButton("✅ Yes")
        self.yes_btn.setStyleSheet("""
            QPushButton{background:#a6e3a1;color:#11111b;border-radius:6px;
                        padding:6px;font-size:11px;font-weight:bold;border:none;}
            QPushButton:hover{background:#c3f0c3;}
            QPushButton:pressed{background:#7ec87e;}""")
        self.no_btn = QPushButton("❌ No")
        self.no_btn.setStyleSheet("""
            QPushButton{background:#f38ba8;color:#11111b;border-radius:6px;
                        padding:6px;font-size:11px;font-weight:bold;border:none;}
            QPushButton:hover{background:#ffb3c6;}
            QPushButton:pressed{background:#e07a96;}""")

        #self.yes_btn.hide()
        #self.no_btn.hide()

        vitals_layout.addWidget(self.yes_btn)
        vitals_layout.addWidget(self.no_btn)

    
        self.question_lbl.hide()
        #self.yes_btn.hide()
        #self.no_btn.hide()

        
        
        sl.addSpacing(10)
        sl.addWidget(vitals_frame)
        self.analyse_btn.hide()
        self.temp_analyse_btn.hide()

        root.addWidget(left_panel)

        # ── RIGHT AREA ────────────────────────────────────────────────────────
        right_widget = QWidget()
        right_widget.setStyleSheet("background:#181825;")
        right_vbox = QVBoxLayout(right_widget)
        right_vbox.setContentsMargins(0, 6, 0, 0)
        right_vbox.setSpacing(8)

        # ── TOP ROW: 3D model (left) + charts (right) ────────────────────────
        top_row = QWidget()
        top_row.setStyleSheet("background:#181825;")
        top_h = QHBoxLayout(top_row)
        top_h.setContentsMargins(0, 0, 0, 0)
        top_h.setSpacing(8)

        # 3D model column
        model_col = QWidget()
        model_col.setStyleSheet("background:#181825;")
        model_col.setFixedWidth(580)
        model_vbox = QVBoxLayout(model_col)
        model_vbox.setContentsMargins(0, 0, 0, 0)
        model_vbox.setSpacing(4)

        hint = QLabel("3D Body Model  —  drag to rotate · scroll to zoom · right-drag to pan")
        hint.setFont(QFont("Arial", 9))
        hint.setStyleSheet("color:#45475a;background:transparent;padding:0px 4px;")
        model_vbox.addWidget(hint)

        self.plotter_host = QWidget()
        self.plotter_host.setStyleSheet("background:#181825;")
        self.plotter_host.setMinimumSize(420, 420)
        self.plotter_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        model_vbox.addWidget(self.plotter_host, stretch=1)

        self.plotter = QtInteractor(self.plotter_host)
        self.plotter.set_background("#181825")

        top_h.addWidget(model_col)  # fixed width, no stretch

        # Charts column (right of 3D model) — fever curve on top, organ risk below
        charts_col = QWidget()
        charts_col.setStyleSheet("background:#181825;")
        charts_vbox = QVBoxLayout(charts_col)
        charts_vbox.setContentsMargins(0, 24, 8, 0)
        charts_vbox.setSpacing(8)

        # Fever curve chart
        self.fever_container = QWidget()
        self.fever_container.setStyleSheet("background:#181825;")
        self.fever_layout = QVBoxLayout(self.fever_container)
        self.fever_layout.setContentsMargins(0, 0, 0, 0)
        charts_vbox.addWidget(self.fever_container, stretch=1)

        # Organ risk chart
        self.risk_container = QWidget()
        self.risk_container.setStyleSheet("background:#181825;")
        self.risk_layout = QVBoxLayout(self.risk_container)
        self.risk_layout.setContentsMargins(0, 0, 0, 0)
        charts_vbox.addWidget(self.risk_container, stretch=1)

        top_h.addWidget(charts_col, stretch=1)
        right_vbox.addWidget(top_row, stretch=1)

        self.draw_placeholder_charts()

        # ── BOTTOM: full-width table ──────────────────────────────────────────
        table_frame = QFrame()
        table_frame.setStyleSheet("QFrame{background:#1a1a2e;border-radius:8px;}")
        table_frame.setMinimumHeight(230)
        table_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_v = QVBoxLayout(table_frame)
        table_v.setContentsMargins(10, 8, 10, 8)
        table_v.setSpacing(4)

        tbl_title = QLabel("🌡  Symptom Progression by Temperature Stage")
        tbl_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        tbl_title.setStyleSheet("color:#cdd6f4;background:transparent;")
        table_v.addWidget(tbl_title)

        self.stage_table = QTableWidget()
        self.stage_table.setColumnCount(3)
        self.stage_table.setHorizontalHeaderLabels(["Temperature Range", "Stage", "Symptoms at This Stage"])
        self.stage_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stage_table.setWordWrap(True)
        self.stage_table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.stage_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stage_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.stage_table.verticalHeader().setVisible(False)
        self.stage_table.setShowGrid(False)
        h = self.stage_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.stage_table.setStyleSheet("""
            QTableWidget{background:#1a1a2e;color:#cdd6f4;border:none;
                         font-size:11px;gridline-color:transparent;outline:none;}
            QTableWidget::item{padding:5px 10px;border-bottom:1px solid #2a2a3e;}
            QHeaderView::section{background:#11111b;color:#6c7086;font-size:10px;
                font-weight:bold;padding:6px 10px;border:none;
                border-bottom:1px solid #313244;letter-spacing:1px;}
            QScrollBar:vertical{background:#1a1a2e;width:5px;border-radius:2px;}
            QScrollBar::handle:vertical{background:#45475a;border-radius:2px;}""")
        self._set_table_placeholder()
        table_v.addWidget(self.stage_table)

        right_vbox.addWidget(table_frame, stretch=1)
        root.addWidget(right_widget, stretch=1)
    def start_final_diagnosis(self):
        try:
            temp = self._parse_temp()
            if temp is None:
                temp_str = self.temp_val_input.text().strip()
                if temp_str and temp_str != "--" and temp_str != "":
                    self.temp_input.setText(temp_str)
                    temp = self._parse_temp()
            
            if temp is None:
                self.checker_result.setPlainText("Please enter a valid temperature first.")
                return

            self.tabs.setCurrentIndex(1)
            
            entry_node = self._get_entry_node(temp)
            node_key, path = self.traverse_tree_with_checkboxes(entry_node)
            
            self._dtree_temp = temp
            self._dtree_path = path
            self._dtree_node = node_key

            node = self.DECISION_TREE.get(node_key)
            if node is None:
                # Reached a leaf (no overlaps, direct result)
                self._show_diagnosis(node_key)
            else:
                # Overlap / unresolved question -> ask the user
                self._show_question()
        except Exception as e:
            print("Decision tree error:", e)

    def is_manual_symptom_active(self, symptom_name):
        # Check checkbox
        if symptom_name in self.symptom_checkboxes and self.symptom_checkboxes[symptom_name].isChecked():
            return True
        # Check text input
        raw_text = self.symptom_input.text().strip()
        if raw_text:
            for token in re.split(r'[,;\n]+', raw_text):
                token = token.strip().lower()
                if token and (token in symptom_name.lower() or symptom_name.lower().split("(")[0].strip() in token):
                    return True
        return False

    def find_matching_symptom(self, question):
        q = question.lower()
        if "rash" in q and "start" not in q:
            return "Skin rash"
        if ("eyes red" in q or "red eyes" in q) and "start" not in q:
            return "Red eyes (Conjunctivitis)"
        if ("jaundice" in q or "yellowing" in q) and "start" not in q:
            return "Jaundice (Yellow skin/eyes)"
        return None

    def traverse_tree_with_checkboxes(self, start_node):
        node_key = start_node
        path = []
        visited = set()
        while True:
            if node_key in visited:
                break
            visited.add(node_key)
            node = self.DECISION_TREE.get(node_key)
            if node is None:
                return node_key, path
            
            question = node["question"]
            matched_sym = self.find_matching_symptom(question)
            if matched_sym is not None:
                is_checked = self.is_manual_symptom_active(matched_sym)
                path.append((question, is_checked))
                node_key = node["yes"] if is_checked else node["no"]
            else:
                return node_key, path
        return node_key, path

    def on_manual_input_changed(self):
        temp = self._parse_temp()
        if temp is None:
            self.checker_result.setPlainText("Enter temperature and tick symptoms to begin...")
            self._restore_normal_buttons()
            return

        self._dtree_temp = temp
        entry_node = self._get_entry_node(temp)
        node_key, path = self.traverse_tree_with_checkboxes(entry_node)
        
        node = self.DECISION_TREE.get(node_key)
        if node is None:
            # Reached a leaf (all symptoms resolved)
            self.checker_result.setPlainText(
                f"🌡  Temperature: {c_to_f(temp):.1f}°F\n"
                f"{'─'*32}\n\n"
                f"All symptoms processed.\n"
                f"Click Final Diagnosis to view the diagnosis."
            )
            self._restore_normal_buttons()
            self._dtree_node = node_key
            self._dtree_path = path
        else:
            self._dtree_node = node_key
            self._dtree_path = path
            self._show_question()
    # ── resize: keep plotter filling its host ────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_plotter()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self._fit_plotter)

    def _fit_plotter(self):
        if hasattr(self, 'plotter') and hasattr(self, 'plotter_host'):
            w = self.plotter_host.width()
            h = self.plotter_host.height()
            if w > 0 and h > 0:
                self.plotter.setGeometry(0, 0, w, h)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _lbl(self, text, bold=False, small=False, color="#cdd6f4"):
        l = QLabel(text)
        fs = "10px" if small else "11px"
        fw = "bold" if bold or small else "normal"
        l.setStyleSheet(f"color:{color};font-size:{fs};font-weight:{fw};background:transparent;")
        return l

    def _clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            w = layout.itemAt(i).widget()
            if w: w.deleteLater()

    def draw_placeholder_charts(self):
        # Fever placeholder
        self._clear_layout(self.fever_layout)
        fig1, ax1 = plt.subplots(figsize=(4, 2.2))
        fig1.patch.set_facecolor("#181825")
        ax1.set_facecolor("#1e1e2e")
        ax1.set_title("Fever Curve (Days)", color="#45475a", fontsize=9, pad=6)
        ax1.text(0.5, 0.5, "Select a disease", color="#313244",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=10)
        for sp in ax1.spines.values(): sp.set_visible(False)
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.tight_layout(pad=0.6)
        c1 = FigureCanvas(fig1)
        c1.setStyleSheet("background:#181825;")
        self.fever_layout.addWidget(c1)
        plt.close(fig1)

        # Risk placeholder
        self._clear_layout(self.risk_layout)
        fig2, ax2 = plt.subplots(figsize=(4, 2.2))
        fig2.patch.set_facecolor("#181825")
        ax2.set_facecolor("#1e1e2e")
        ax2.set_title("Organ Risk Level", color="#45475a", fontsize=9, pad=6)
        ax2.text(0.5, 0.5, "Select a disease", color="#313244",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        for sp in ax2.spines.values(): sp.set_visible(False)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.tight_layout(pad=0.6)
        c2 = FigureCanvas(fig2)
        c2.setStyleSheet("background:#181825;")
        self.risk_layout.addWidget(c2)
        plt.close(fig2)

    # ── DISEASE CHARTS ────────────────────────────────────────────────────────
    def draw_disease_charts(self, name):
        # ── Fever Curve ───────────────────────────────────────────────────────
        self._clear_layout(self.fever_layout)
        fig1, ax1 = plt.subplots(figsize=(4, 2.2))
        fig1.patch.set_facecolor("#181825")
        ax1.set_facecolor("#1e1e2e")
        curve = FEVER_CURVES.get(name, [])
        if curve:
            days  = [p[0] for p in curve]
            temps = [p[1] for p in curve]   # already in °F
            normal_f = c_to_f(37.0)
            for lo, hi, col in [
                (c_to_f(37.0), c_to_f(37.5), "#a6e3a1"),
                (c_to_f(37.5), c_to_f(38.5), "#f9e2af"),
                (c_to_f(38.5), c_to_f(39.5), "#fab387"),
                (c_to_f(39.5), c_to_f(42.0), "#f38ba8")]:
                ax1.axhspan(lo, hi, color=col, alpha=0.12)
            ax1.plot(days, temps, color="#ff6b6b", linewidth=2, zorder=3)
            ax1.fill_between(days, normal_f, temps, alpha=0.15, color="#ff6b6b")
            ax1.scatter(days, temps, color="#ff2d55", s=28, zorder=4)
            ax1.axhline(normal_f, color="#45475a", linewidth=0.8, linestyle="--")
            ax1.text(days[-1]+0.05, normal_f+0.1, "Normal 98.6°F",
                     color="#45475a", fontsize=7, va="bottom")
            peak_day = days[temps.index(max(temps))]
            peak_t   = max(temps)
            ax1.annotate(f"Peak {peak_t}°F",
                         xy=(peak_day, peak_t),
                         xytext=(peak_day + 0.5, peak_t - 0.7),
                         color="#ff2d55", fontsize=7,
                         arrowprops=dict(arrowstyle="->", color="#ff2d55", lw=0.8))
            ax1.set_xlim(days[0]-0.3, days[-1]+1.0)
            ax1.set_ylim(normal_f - 1.0, max(temps) + 1.2)
            ax1.set_xlabel("Day of illness", color="#6c7086", fontsize=8)
            ax1.set_ylabel("Temp (°F)", color="#6c7086", fontsize=8)
            ax1.tick_params(colors="#6c7086", labelsize=7)
        ax1.set_title(f"{name} — Fever Progression", color="#cdd6f4", fontsize=9, pad=5)
        for sp in ax1.spines.values(): sp.set_color("#313244")
        plt.tight_layout(pad=0.6)
        c1 = FigureCanvas(fig1)
        c1.setStyleSheet("background:#181825;")
        self.fever_layout.addWidget(c1)
        plt.close(fig1)

        # ── Organ Risk ────────────────────────────────────────────────────────
        self._clear_layout(self.risk_layout)
        fig2, ax2 = plt.subplots(figsize=(4, 2.2))
        fig2.patch.set_facecolor("#181825")
        ax2.set_facecolor("#1e1e2e")
        risk = ORGAN_RISK.get(name, {})
        if risk:
            organs = [o.replace("_"," ").title() for o in risk]
            values = list(risk.values())
            bar_colors = ["#ff2d55" if v>=9 else "#fab387" if v>=7
                          else "#f9e2af" if v>=5 else "#a6e3a1" for v in values]
            bars = ax2.barh(organs, values, color=bar_colors, edgecolor="none", height=0.55)
            for bar, val in zip(bars, values):
                ax2.text(val+0.15, bar.get_y()+bar.get_height()/2,
                         f"{val}/10", va="center", color="#cdd6f4", fontsize=7)
            ax2.set_xlim(0, 11)
            ax2.set_xlabel("Risk Level", color="#6c7086", fontsize=8)
            ax2.tick_params(colors="#6c7086", labelsize=7.5)
            ax2.axvline(5, color="#313244", linewidth=0.6, linestyle="--")
            ax2.axvline(8, color="#45475a", linewidth=0.6, linestyle="--")
            legend = [
                mpatches.Patch(color="#ff2d55", label="Critical (9-10)"),
                mpatches.Patch(color="#fab387", label="High (7-8)"),
                mpatches.Patch(color="#f9e2af", label="Moderate (5-6)"),
                mpatches.Patch(color="#a6e3a1", label="Low (<5)"),
            ]
            ax2.legend(handles=legend, loc="lower left",
                       bbox_to_anchor=(0, 1.01), ncol=2,
                       fontsize=6, framealpha=0.3, labelcolor="#cdd6f4",
                       facecolor="#1e1e2e", edgecolor="#313244",
                       borderpad=0.5, handlelength=1.0)
        ax2.set_title(f"{name} — Organ Risk Index", color="#cdd6f4", fontsize=9, pad=5)
        for sp in ax2.spines.values(): sp.set_color("#313244")
        plt.tight_layout(pad=0.6)
        c2 = FigureCanvas(fig2)
        c2.setStyleSheet("background:#181825;")
        self.risk_layout.addWidget(c2)
        plt.close(fig2)

    # ── THERMOMETER ───────────────────────────────────────────────────────────
    def draw_thermometer(self, temperature_c):
        for i in reversed(range(self.temp_layout.count())):
            w = self.temp_layout.itemAt(i).widget()
            if w: w.deleteLater()

        temperature = c_to_f(temperature_c)   # convert to °F for display
        normal_f, max_f = 98.6, 107.6         # 37°C, 42°C in °F
        frac = max(0, min((temperature - normal_f) / (max_f - normal_f), 1))

        if temperature_c <= 37.5:   color = "#a6e3a1"
        elif temperature_c <= 38.5: color = "#f9e2af"
        elif temperature_c <= 39.5: color = "#fab387"
        else:                       color = "#f38ba8"

        fig, ax = plt.subplots(figsize=(2.6, 1.4))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#2a2a3e")

        ax.barh(0, 1,    color="#313244", height=0.5, edgecolor="none")
        ax.barh(0, frac, color=color,    height=0.5, edgecolor="none")

        # Tick marks at key °F values
        for tc, lbl in [(37,"99.5°"),(38,"100.4°"),(39,"102.2°"),(40,"104°"),(41,"105.8°")]:
            tf = c_to_f(tc)
            x  = (tf - normal_f) / (max_f - normal_f)
            ax.axvline(x, color="#6c7086", linewidth=0.6, ymin=0.1, ymax=0.9)
            ax.text(x, -0.45, lbl, color="#6c7086", fontsize=5.5, ha="center", va="top")

        ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([]); ax.set_xticks([])
        ax.text(0.5, 1.18, f"{temperature}°F", color=color,
                ha="center", va="bottom", fontsize=15,
                fontweight="bold", transform=ax.transAxes)
        for sp in ax.spines.values(): sp.set_visible(False)

        plt.tight_layout(pad=0.3)
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(100)
        canvas.setStyleSheet("background:transparent;")
        self.temp_layout.addWidget(canvas)
        plt.close(fig)

    # ── TABLE ─────────────────────────────────────────────────────────────────
    def _set_table_placeholder(self):
        self.stage_table.setRowCount(1)
        p = QTableWidgetItem("← Select a disease")
        p.setForeground(QColor("#45475a"))
        p.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.stage_table.setItem(0, 0, QTableWidgetItem(""))
        self.stage_table.setItem(0, 1, QTableWidgetItem(""))
        self.stage_table.setItem(0, 2, p)

    def populate_stage_table(self, stages):
        self.stage_table.setRowCount(len(stages))
        for row, entry in enumerate(stages):
            stage_color = STAGE_COLORS.get(entry["stage"], "#cdd6f4")
            bg = QColor("#1a1a2e") if row % 2 == 0 else QColor("#1f1f35")

            # Convert °C range text to °F  e.g. "37.0°C – 38.0°C" → "98.6°F – 100.4°F"
            def convert_range(rng):
                def rep(m):
                    val = float(m.group(1))
                    return f"{c_to_f(val):.1f}°F"
                return re.sub(r'([\d.]+)°C', rep, rng)
            converted_range = convert_range(entry["range"])

            t = QTableWidgetItem(f"  {converted_range}")
            t.setForeground(QColor("#89b4fa"))
            t.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            t.setBackground(bg); t.setFlags(Qt.ItemFlag.ItemIsEnabled)

            s = QTableWidgetItem(f"  {entry['stage']}")
            s.setForeground(QColor(stage_color))
            s.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            s.setBackground(bg); s.setFlags(Qt.ItemFlag.ItemIsEnabled)

            # Strip citation references like [CDC ...] from display text
            clean_sym = re.sub(r'\s*\[.*?\]', '', entry["symptoms"]).strip()

            sym = QTableWidgetItem(clean_sym)
            sym.setForeground(QColor("#cdd6f4"))
            sym.setFont(QFont("Arial", 10))
            sym.setBackground(bg); sym.setFlags(Qt.ItemFlag.ItemIsEnabled)

            self.stage_table.setItem(row, 0, t)
            self.stage_table.setItem(row, 1, s)
            self.stage_table.setItem(row, 2, sym)

        self.stage_table.resizeRowsToContents()

    # ── ORGAN MESHES ──────────────────────────────────────────────────────────
    ORGAN_MESHES = {
        "brain":       lambda: pv.Sphere(radius=0.12, center=(0,     0.78,  0.05)),
        "eyes":        lambda: pv.Sphere(radius=0.04, center=(0.07,  0.82,  0.16)),
        "heart":       lambda: pv.Sphere(radius=0.07, center=(-0.08, 0.28,  0.16)),
        "lymph_nodes": lambda: pv.Sphere(radius=0.05, center=(0,     0.42,  0.16)),
        "liver":       lambda: pv.Sphere(radius=0.09, center=(0.10,  0.14,  0.16)),
        "spleen":      lambda: pv.Sphere(radius=0.07, center=(-0.14, 0.05,  0.16)),
        "blood":       lambda: pv.Sphere(radius=0.05, center=(0,     0.10,  0.20)),
        "muscles":     lambda: pv.Sphere(radius=0.06, center=(0.18,  0.00,  0.16)),
        "kidney":      lambda: pv.Sphere(radius=0.06, center=(0.12, -0.06,  0.16)),
        "spinal_cord": lambda: pv.Cylinder(radius=0.03, height=1.0, center=(0,0,-0.10), direction=(0,1,0)),
        "skin":        lambda: pv.Sphere(radius=0.06, center=(0.25,  0.00,  0.00)),
        "joints":      lambda: pv.Sphere(radius=0.05, center=(0,    -0.50,  0.00)),
        "limbs":       lambda: pv.Cylinder(radius=0.08, height=0.85, center=(0.13,-0.93,0), direction=(0,1,0)),
    }

    ORGAN_LABELS = {
        "brain":       ((0,     0.78, 0.05), "Brain",       ( 0.60,  0.95, 0.05)),
        "eyes":        ((0.07,  0.82, 0.16), "Eyes",        ( 0.60,  1.10, 0.05)),
        "heart":       ((-0.08, 0.28, 0.16), "Heart",       (-0.62,  0.50, 0.05)),
        "lymph_nodes": ((0,     0.42, 0.16), "Lymph Nodes", ( 0.62,  0.65, 0.05)),
        "liver":       ((0.10,  0.14, 0.16), "Liver",       ( 0.62,  0.20, 0.05)),
        "spleen":      ((-0.14, 0.05, 0.16), "Spleen",      (-0.62,  0.10, 0.05)),
        "blood":       ((0,     0.10, 0.20), "Blood",       ( 0.62, -0.05, 0.05)),
        "muscles":     ((0.18,  0.00, 0.16), "Muscles",     ( 0.62, -0.22, 0.05)),
        "kidney":      ((0.12, -0.06, 0.16), "Kidney",      (-0.62, -0.10, 0.05)),
        "spinal_cord": ((0,     0.00,-0.10), "Spinal Cord", (-0.62, -0.28, 0.05)),
        "skin":        ((0.25,  0.00, 0.00), "Skin",        ( 0.62, -0.40, 0.05)),
        "joints":      ((0,    -0.50, 0.00), "Joints",      (-0.62, -0.55, 0.05)),
        "limbs":       ((0.13, -0.93, 0.00), "Limbs",       ( 0.62, -0.93, 0.05)),
    }

    def load_body_model(self, affected=None):
        affected = affected or []
        self.plotter.clear()
        self.plotter.set_background("#181825")

        for mesh in [
            pv.Cylinder(radius=0.25, height=1.2, center=(0,0,0),        direction=(0,1,0)),
            pv.Sphere  (radius=0.18,              center=(0,0.78,0)),
            pv.Cylinder(radius=0.07, height=0.85, center=(-0.4,0.1,0),  direction=(0.2,1,0)),
            pv.Cylinder(radius=0.07, height=0.85, center=(0.4, 0.1,0),  direction=(-0.2,1,0)),
            pv.Cylinder(radius=0.09, height=0.9,  center=(-0.13,-0.95,0),direction=(0,1,0)),
            pv.Cylinder(radius=0.09, height=0.9,  center=(0.13,-0.95,0), direction=(0,1,0)),
        ]:
            self.plotter.add_mesh(mesh, color="#89b4fa", opacity=0.40, smooth_shading=True)

        for name, mesh_fn in self.ORGAN_MESHES.items():
            if name in affected:
                self.plotter.add_mesh(mesh_fn(), color=AFFECTED_COLOR, opacity=1.0, smooth_shading=True)

        for name, (organ_pos, label, label_pos) in self.ORGAN_LABELS.items():
            ia = name in affected
            self.plotter.add_mesh(pv.Line(organ_pos, label_pos),
                                  color=AFFECTED_COLOR if ia else "#3a3a5c",
                                  line_width=2.0 if ia else 0.8)
            self.plotter.add_point_labels(
                [label_pos], [f"  {label}  ◄" if ia else f"  {label}"],
                font_size=12 if ia else 9,
                text_color=AFFECTED_COLOR if ia else "#555577",
                point_color=AFFECTED_COLOR if ia else "#3a3a5c",
                point_size=7 if ia else 3,
                render_points_as_spheres=True,
                always_visible=True, shadow=False, shape=None,
            )

        self.plotter.enable_trackball_style()
        self.plotter.view_vector((0, 0, 1), viewup=(0, 1, 0))
        self.plotter.camera.zoom(0.7)
        self.plotter.render()
        self._fit_plotter()

    # ── DECISION TREE ─────────────────────────────────────────────────────────
    # Each node: {"question": str, "yes": disease_name OR next_node_key, "no": ...}
    # Leaf nodes are disease names (strings matching self.diseases keys)
    # Built from clinical differentiators between overlapping fever diseases

    DECISION_TREE = {
        # ── Entry by temperature zone ──────────────────────────────────────────
        "zone_mild": {          # 37.5–38.5°C  →  Zika vs West Nile vs Chikungunya vs Filariasis
            "question": "Do you have a skin rash?",
            "yes": "zone_mild_rash",
            "no":  "zone_mild_no_rash",
        },
        "zone_mild_rash": {     # rash present
            "question": "Are your eyes red / irritated (conjunctivitis)?",
            "yes": "Zika",
            "no":  "zone_mild_rash_no_eye",
        },
        "zone_mild_rash_no_eye": {
            "question": "Do you have severe joint pain and swelling?",
            "yes": "Chikungunya",
            "no":  "West Nile Fever",
        },
        "zone_mild_no_rash": {  # no rash
            "question": "Do you have swollen limbs or swollen lymph nodes?",
            "yes": "zone_mild_swelling",
            "no":  "zone_mild_no_rash_no_swell",
        },
        "zone_mild_swelling": {
            "question": "Is the swelling mainly in your legs / arms (not lymph nodes)?",
            "yes": "Lymphatic Filariasis",
            "no":  "West Nile Fever",
        },
        "zone_mild_no_rash_no_swell": {
            "question": "Do you have neck stiffness or confusion?",
            "yes": "West Nile Fever",
            "no":  "Zika",
        },

        # ── 38.5–39.5°C  →  Chikungunya vs West Nile vs Filariasis vs Dengue ──
        "zone_moderate": {
            "question": "Do you have severe joint pain and swelling in multiple joints?",
            "yes": "zone_mod_joints",
            "no":  "zone_mod_no_joints",
        },
        "zone_mod_joints": {
            "question": "Did the joint pain start within 1–2 days of the fever?",
            "yes": "Chikungunya",
            "no":  "zone_mod_joints_late",
        },
        "zone_mod_joints_late": {
            "question": "Do you also have bleeding gums or easy bruising?",
            "yes": "Dengue",
            "no":  "Chikungunya",
        },
        "zone_mod_no_joints": {
            "question": "Do you have swollen limbs (legs / arms getting bigger)?",
            "yes": "Lymphatic Filariasis",
            "no":  "zone_mod_no_joints_no_limb",
        },
        "zone_mod_no_joints_no_limb": {
            "question": "Do you have neck stiffness or neurological symptoms (confusion, seizures)?",
            "yes": "West Nile Fever",
            "no":  "zone_mod_check_dengue",
        },
        "zone_mod_check_dengue": {
            "question": "Do you have pain behind the eyes or severe headache?",
            "yes": "Dengue",
            "no":  "West Nile Fever",
        },

        # ── 39.5–40.5°C  →  Dengue vs Yellow Fever vs Malaria ────────────────
        "zone_high": {
            "question": "Do you have yellowing of the skin or eyes (jaundice)?",
            "yes": "zone_high_jaundice",
            "no":  "zone_high_no_jaundice",
        },
        "zone_high_jaundice": {
            "question": "Did the fever seem to improve briefly before getting worse again?",
            "yes": "Yellow Fever",
            "no":  "zone_high_jaundice_check",
        },
        "zone_high_jaundice_check": {
            "question": "Do you have severe chills (bone-rattling rigors) cyclically?",
            "yes": "Malaria",
            "no":  "Yellow Fever",
        },
        "zone_high_no_jaundice": {
            "question": "Do you have severe pain behind the eyes and bleeding gums / bruising?",
            "yes": "Dengue",
            "no":  "zone_high_no_jaundice_check",
        },
        "zone_high_no_jaundice_check": {
            "question": "Do you have cyclic chills (repeating every 1–2 days)?",
            "yes": "Malaria",
            "no":  "zone_high_final",
        },
        "zone_high_final": {
            "question": "Do you have an enlarged spleen or anaemia?",
            "yes": "Malaria",
            "no":  "Dengue",
        },

        # ── 40.5°C+  →  Malaria vs Yellow Fever ──────────────────────────────
        "zone_critical": {
            "question": "Do you have yellowing of skin/eyes and are vomiting blood?",
            "yes": "Yellow Fever",
            "no":  "zone_crit_check",
        },
        "zone_crit_check": {
            "question": "Do you have cyclic rigors/chills and confusion or seizures?",
            "yes": "Malaria",
            "no":  "zone_crit_final",
        },
        "zone_crit_final": {
            "question": "Do you have severe bleeding from mouth, nose or gums?",
            "yes": "Yellow Fever",
            "no":  "Malaria",
        },
    }

    def _parse_temp(self):
        """Parse °F input, validate range 95–115°F, return as °C float or None."""
        raw = self.temp_input.text().strip().replace(",", ".")
        try:
            f = float(raw)
            if 95.0 <= f <= 115.0:
                return f_to_c(f)   # convert to °C for internal logic
        except ValueError:
            pass
        return None

    DISEASE_TEMP_RANGES = {   # (min_°F, peak_°F, max_°F)
        "Malaria":              (c_to_f(38.0), c_to_f(40.5), c_to_f(41.5)),
        "Dengue":               (c_to_f(38.0), c_to_f(40.0), c_to_f(41.0)),
        "Chikungunya":          (c_to_f(38.0), c_to_f(39.5), c_to_f(40.5)),
        "Zika":                 (c_to_f(37.5), c_to_f(38.5), c_to_f(39.0)),
        "Yellow Fever":         (c_to_f(38.5), c_to_f(40.0), c_to_f(41.5)),
        "West Nile Fever":      (c_to_f(37.5), c_to_f(39.0), c_to_f(40.5)),
        "Lymphatic Filariasis": (c_to_f(38.0), c_to_f(38.8), c_to_f(39.5)),
    }

    def _temp_score(self, disease, temp):
        """Triangular proximity score 0–1 for how well temp fits disease range."""
        lo, peak, hi = self.DISEASE_TEMP_RANGES[disease]
        if temp < lo or temp > hi:
            dist = min(abs(temp - lo), abs(temp - hi))
            return max(0.0, 1.0 - dist * 0.5)
        if temp <= peak:
            return (temp - lo) / (peak - lo) if peak != lo else 1.0
        else:
            return (hi - temp) / (hi - peak) if hi != peak else 1.0

    def _format_result(self, ranked, temp=None, mode="combined"):
        """Build result text for the combined symptom checker."""
        top_disease, top_data = ranked[0]
        sev_icons = {"Mild":"🟢","Moderate":"🟡","High":"🟠","Critical":"🔴","Chronic":"🟣"}
        sev  = self.diseases[top_disease]["severity"]
        mosq = self.diseases[top_disease]["mosquito"]
        icon = sev_icons.get(sev, "⚪")
        lines = [f"🔬  COMBINED ANALYSIS\n{'─'*32}"]
        if temp:
            lines.append(f"\n   Temperature: {c_to_f(temp):.1f}°F")
        lines.append(f"\n🏆  Most Likely:  {top_disease}")
        lines.append(f"    Confidence:  {top_data['score']}%")
        lines.append(f"    {icon} Severity:  {sev}")
        lines.append(f"    🦟 Mosquito:  {mosq}")
        if top_data.get("matches"):
            lines.append(f"    ✔ Matched:  {', '.join(top_data['matches'][:3])}")
        if len(ranked) > 1:
            lines.append(f"\n📋  Other Possibilities:")
            for name, data in ranked[1:4]:
                lines.append(f"    • {name}  ({data['score']}%)")
        lines.append(f"\n💡  Loading '{top_disease}' in Disease tab…")
        lines.append(f"\n{'─'*32}")
        lines.append("⚕  Educational only. Consult a doctor.")
        return "\n".join(lines)

    # Map temperature to entry node
    def _get_entry_node(self, temp_c):
        """temp_c is in Celsius internally."""
        if temp_c < 38.5:
            return "zone_mild"
        elif temp_c < 39.5:
            return "zone_moderate"
        elif temp_c < 40.5:
            return "zone_high"
        else:
            return "zone_critical"

    def run_temperature_check(self):
        """Step 1 — temperature entry → start decision tree."""
        temp = self._parse_temp()
        if temp is None:
            self.checker_result.setPlainText(
                "⚠  Please enter a valid temperature\n"
                "   between 95.0°F and 115.0°F\n\n"
                "   Example:  103.1")
            return

        # Store temp and start decision tree
        self._dtree_temp   = temp
        self._dtree_path   = []   # list of (question, answer) pairs
        self._dtree_node   = self._get_entry_node(temp)
        self._show_question()

    def _show_question(self):
        self.question_lbl.show()
        self.yes_btn.show()
        self.no_btn.show()
        """Render the current decision tree question with Yes/No buttons."""
        node = self.DECISION_TREE.get(self._dtree_node)
        if node is None:
            # _dtree_node is a disease name — we've reached a leaf
            self._show_diagnosis(self._dtree_node)
            return

        question = node["question"]

        # Clear result area and show the question + path so far
        lines = [f"🌡  Temperature: {c_to_f(self._dtree_temp):.1f}°F\n{'─'*32}"]
        if self._dtree_path:
            lines.append("\nAnswered so far:")
            for i, (q, a) in enumerate(self._dtree_path, 1):
                lines.append(f"  Q{i}: {q}")
                lines.append(f"      → {'✅ Yes' if a else '❌ No'}")
        lines.append(f"\n{'─'*32}")
        lines.append(f"❓  {question}")
        self.checker_result.setPlainText("\n".join(lines))

        # Show Yes / No buttons (replace analyse btn row temporarily)
        self._show_yn_buttons(question, node)

    def _show_yn_buttons(self, question, node):
        """Replace the analyse button with Yes/No for this question."""
        # Hide normal buttons
        self.temp_analyse_btn.setVisible(False)
        self.analyse_btn.setVisible(False)

        # Create yes/no button row if not exists
        if not hasattr(self, "_yn_widget"):
            self._yn_widget = QWidget()
            self._yn_widget.setStyleSheet("background:transparent;")
            yn_row = QHBoxLayout(self._yn_widget)
            yn_row.setContentsMargins(0, 0, 0, 0)
            yn_row.setSpacing(8)

            self._yes_btn = QPushButton("✅  Yes")
            self._yes_btn.setStyleSheet("""
                QPushButton{background:#a6e3a1;color:#1e1e2e;border-radius:7px;
                            padding:8px;font-size:12px;font-weight:bold;border:none;}
                QPushButton:hover{background:#c3f0c3;}""")

            self._no_btn = QPushButton("❌  No")
            self._no_btn.setStyleSheet("""
                QPushButton{background:#f38ba8;color:#1e1e2e;border-radius:7px;
                            padding:8px;font-size:12px;font-weight:bold;border:none;}
                QPushButton:hover{background:#ffb3c6;}""")

            self._back_btn = QPushButton("← Back")
            self._back_btn.setStyleSheet("""
                QPushButton{background:#313244;color:#6c7086;border-radius:7px;
                            padding:8px;font-size:11px;border:none;}
                QPushButton:hover{background:#45475a;color:#cdd6f4;}""")

            yn_row.addWidget(self._yes_btn)
            yn_row.addWidget(self._no_btn)
            yn_row.addWidget(self._back_btn)

            # Insert before checker_result in the layout
            parent_layout = self.checker_result.parent().layout()
            idx = parent_layout.indexOf(self.checker_result)
            parent_layout.insertWidget(idx, self._yn_widget)

        self._yn_widget.setVisible(True)

        # Disconnect old signals safely
        try: self._yes_btn.clicked.disconnect()
        except: pass
        try: self._no_btn.clicked.disconnect()
        except: pass
        try: self._back_btn.clicked.disconnect()
        except: pass

        self._yes_btn.clicked.connect(lambda: self._answer(True,  node))
        self._no_btn.clicked.connect( lambda: self._answer(False, node))
        self._back_btn.clicked.connect(self._go_back)

    def _answer(self, yes, node):
        """Process a Yes/No answer and advance the tree."""
        self._dtree_path.append((node["question"], yes))
        next_node = node["yes"] if yes else node["no"]

        if next_node in self.diseases:
            # Reached a leaf — diagnosis done
            self._dtree_node = next_node
            self._show_diagnosis(next_node)
        else:
            self._dtree_node = next_node
            self._show_question()

    def _go_back(self):
        """Go back one question in the decision tree."""
        if not self._dtree_path:
            # Back to start
            self._restore_normal_buttons()
            self.checker_result.setPlainText(
                "Enter temperature, tick symptoms, and click Final Diagnosis.")
            return
        self._dtree_path.pop()
        # Replay path from entry node to get current node
        node_key = self._get_entry_node(self._dtree_temp)
        for _, answered_yes in self._dtree_path:
            n = self.DECISION_TREE[node_key]
            node_key = n["yes"] if answered_yes else n["no"]
        self._dtree_node = node_key
        self._show_question()

    def _show_diagnosis(self, disease):
        """Final diagnosis display after decision tree completes."""
        self._restore_normal_buttons()

        sev  = self.diseases[disease]["severity"]
        mosq = self.diseases[disease]["mosquito"]
        organs = ", ".join(o.replace("_"," ").title()
                           for o in self.diseases[disease]["organs"])
        sev_icons = {"Mild":"🟢","Moderate":"🟡","High":"🟠","Critical":"🔴","Chronic":"🟣"}
        icon = sev_icons.get(sev, "⚪")

        path_lines = []
        for i, (q, a) in enumerate(self._dtree_path, 1):
            path_lines.append(f"  Q{i}: {q}")
            path_lines.append(f"      → {'✅ Yes' if a else '❌ No'}")

        lines = [
            f"🌡  Temperature: {c_to_f(self._dtree_temp):.1f}°F",
            f"{'─'*32}",
            "",
            f"🏆  DIAGNOSIS:  {disease}",
            f"    {icon} Severity:  {sev}",
            f"    🦟 Mosquito:  {mosq}",
            f"    🫀 Organs:    {organs}",
            "",
            "📋  Your answers:",
        ] + path_lines + [
            "",
            f"💡  Loading '{disease}' in Disease tab…",
            f"{'─'*32}",
            "⚕  Educational only. Consult a doctor.",
        ]
        self.checker_result.setPlainText("\n".join(lines))

        # Auto-load diagnosed disease
        self.combo.setCurrentText(disease)
        self.tabs.setCurrentIndex(0)

    def _restore_normal_buttons(self):
        """Hide Yes/No row."""
        if hasattr(self, "_yn_widget"):
            self._yn_widget.setVisible(False)

    def run_symptom_checker(self):
        """Step 2 — temperature + symptoms combined."""
        temp = self._parse_temp()

        # Collect checked symptoms
        checked = [sym for sym, cb in self.symptom_checkboxes.items() if cb.isChecked()]

        # Parse free text
        raw_text = self.symptom_input.text().strip()
        text_syms = []
        if raw_text:
            for token in re.split(r'[,;\n]+', raw_text):
                token = token.strip().lower()
                if token:
                    for known in ALL_SYMPTOMS:
                        if token in known.lower() or \
                           known.lower().split("(")[0].strip() in token:
                            if known not in text_syms:
                                text_syms.append(known)

        all_selected = list(set(checked + text_syms))

        if not all_selected and temp is None:
            self.checker_result.setPlainText(
                "⚠  Please enter a temperature and/or\n"
                "   select at least one symptom.")
            return

        scores = {}
        for disease, dsyms in DISEASE_SYMPTOMS.items():
            # ── Symptom F1 score ──────────────────────────────────────────────
            if all_selected:
                matches   = [s for s in all_selected if s in dsyms]
                if matches:
                    precision = len(matches) / len(all_selected)
                    recall    = len(matches) / len(dsyms)
                    f1        = (2 * precision * recall / (precision + recall))
                else:
                    f1      = 0.0
                    matches = []
            else:
                f1      = 0.5   # no symptoms entered — neutral
                matches = []

            # ── Temperature score ─────────────────────────────────────────────
            if temp is not None:
                ts = self._temp_score(disease, temp)
            else:
                ts = 0.5        # no temperature — neutral

            # ── Combined score: 50% temp + 50% symptoms ───────────────────────
            combined = (ts * 0.5) + (f1 * 0.5)
            scores[disease] = {
                "score":   round(combined * 100),
                "matches": matches,
            }

        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        ranked = [(d, s) for d, s in ranked if s["score"] > 10]

        if not ranked:
            self.checker_result.setPlainText(
                "❌  No strong match found.\n\n"
                "   Try adding more symptoms or\n"
                "   verify the temperature entered.")
            return

        self.checker_result.setPlainText(
            self._format_result(ranked, temp=temp, mode="combined"))

        top = ranked[0][0]
        self.combo.setCurrentText(top)
        self.tabs.setCurrentIndex(0)

    def clear_checker(self):
        for cb in self.symptom_checkboxes.values():
            cb.setChecked(False)
        self.symptom_input.clear()
        self.temp_input.clear()
        self.travel_input.setCurrentIndex(0)
        self.checker_result.clear()
        self._dtree_path = []
        self._dtree_node = None
        self._dtree_temp = None
        self._restore_normal_buttons()

    def get_possible_diseases(self, temp, hr, spo2, bp_sys, bp_dia):
        # 1. Priors based on travel history
        priors = {
            "Malaria": 0.02,
            "Dengue": 0.02,
            "Chikungunya": 0.02,
            "Zika": 0.02,
            "Yellow Fever": 0.02,
            "West Nile Fever": 0.02,
            "Lymphatic Filariasis": 0.02,
        }
        
        region = self.travel_input.currentData() if hasattr(self, 'travel_input') else "none"
        if region == "africa":
            priors = {
                "Malaria": 0.45,
                "Yellow Fever": 0.08,
                "Dengue": 0.05,
                "Chikungunya": 0.05,
                "Zika": 0.02,
                "West Nile Fever": 0.05,
                "Lymphatic Filariasis": 0.05,
            }
        elif region == "asia":
            priors = {
                "Dengue": 0.40,
                "Chikungunya": 0.15,
                "Malaria": 0.10,
                "West Nile Fever": 0.05,
                "Zika": 0.05,
                "Lymphatic Filariasis": 0.05,
                "Yellow Fever": 0.01,
            }
        elif region == "americas":
            priors = {
                "Zika": 0.25,
                "Dengue": 0.30,
                "Chikungunya": 0.15,
                "West Nile Fever": 0.05,
                "Malaria": 0.02,
                "Yellow Fever": 0.02,
                "Lymphatic Filariasis": 0.02,
            }
            
        # 2. Temperature likelihood
        temp_lh = {
            "Malaria": 0.90 if temp >= 104.0 else (0.60 if temp >= 102.0 else (0.20 if temp >= 99.5 else 0.05)),
            "Dengue": 0.85 if temp >= 103.0 else (0.50 if temp >= 101.0 else (0.20 if temp >= 99.5 else 0.05)),
            "Chikungunya": 0.90 if temp >= 102.0 else (0.40 if temp >= 100.0 else (0.15 if temp >= 99.0 else 0.05)),
            "Zika": 0.80 if 99.0 <= temp <= 101.5 else (0.40 if 101.5 < temp <= 103.0 else (0.10 if temp > 103.0 else 0.15)),
            "Yellow Fever": 0.80 if temp >= 103.0 else (0.40 if temp >= 101.0 else (0.15 if temp >= 99.0 else 0.05)),
            "West Nile Fever": 0.75 if 99.5 <= temp <= 102.0 else (0.30 if 102.0 < temp <= 104.0 else (0.10 if temp > 104.0 else 0.20)),
            "Lymphatic Filariasis": 0.70 if temp < 100.5 else (0.30 if temp <= 102.0 else 0.05)
        }

        # 3. Heart rate likelihood (incorporating Faget's sign for Yellow Fever)
        hr_lh = {}
        if temp >= 102.0 and hr < 90:
            hr_lh["Yellow Fever"] = 0.80
            for d in priors:
                if d != "Yellow Fever":
                    hr_lh[d] = 0.15
        else:
            if hr > 110:
                hr_lh = {
                    "Malaria": 0.75,
                    "Dengue": 0.70,
                    "Yellow Fever": 0.50,
                    "Chikungunya": 0.60,
                    "Zika": 0.40,
                    "West Nile Fever": 0.40,
                    "Lymphatic Filariasis": 0.30
                }
            else:
                hr_lh = {d: 0.50 for d in priors}

        # 4. SpO2 likelihood
        spo2_lh = {}
        if spo2 < 90:
            spo2_lh = {
                "Yellow Fever": 0.50,
                "Malaria": 0.45,
                "West Nile Fever": 0.35,
                "Dengue": 0.20,
                "Chikungunya": 0.10,
                "Zika": 0.10,
                "Lymphatic Filariasis": 0.15
            }
        elif spo2 < 94:
            spo2_lh = {
                "Yellow Fever": 0.40,
                "Malaria": 0.35,
                "West Nile Fever": 0.30,
                "Dengue": 0.25,
                "Chikungunya": 0.15,
                "Zika": 0.15,
                "Lymphatic Filariasis": 0.20
            }
        else:
            spo2_lh = {d: 0.80 for d in priors}

        # 5. Blood pressure likelihood
        bp_lh = {}
        if bp_sys < 90 or bp_dia < 60:
            bp_lh = {
                "Dengue": 0.75,
                "Yellow Fever": 0.70,
                "Malaria": 0.35,
                "Chikungunya": 0.15,
                "Zika": 0.15,
                "West Nile Fever": 0.15,
                "Lymphatic Filariasis": 0.10
            }
        elif bp_sys < 100 or bp_dia < 70:
            bp_lh = {
                "Dengue": 0.50,
                "Yellow Fever": 0.45,
                "Malaria": 0.30,
                "Chikungunya": 0.25,
                "Zika": 0.25,
                "West Nile Fever": 0.25,
                "Lymphatic Filariasis": 0.20
            }
        else:
            bp_lh = {d: 0.80 for d in priors}

        # 6. Camera symptoms likelihood
        camera_lh = {d: 1.0 for d in priors}
        if self.red_eye_detected:
            for d in priors:
                if d == "Zika":
                    camera_lh[d] *= 0.70
                elif d == "Dengue":
                    camera_lh[d] *= 0.15
                else:
                    camera_lh[d] *= 0.05
        else:
            for d in priors:
                if d == "Zika":
                    camera_lh[d] *= 0.30
                elif d == "Dengue":
                    camera_lh[d] *= 0.85
                else:
                    camera_lh[d] *= 0.95

        if self.rash_detected:
            for d in priors:
                if d == "Zika":
                    camera_lh[d] *= 0.80
                elif d in ["Dengue", "Chikungunya"]:
                    camera_lh[d] *= 0.50
                elif d == "West Nile Fever":
                    camera_lh[d] *= 0.25
                else:
                    camera_lh[d] *= 0.05
        else:
            for d in priors:
                if d == "Zika":
                    camera_lh[d] *= 0.20
                elif d in ["Dengue", "Chikungunya"]:
                    camera_lh[d] *= 0.50
                elif d == "West Nile Fever":
                    camera_lh[d] *= 0.75
                else:
                    camera_lh[d] *= 0.95

        # 7. Live symptoms feedback likelihood
        live_lh = {d: 1.0 for d in priors}
        for symptom, value in self.live_symptoms_responses.items():
            for d in priors:
                dsyms = DISEASE_SYMPTOMS.get(d, [])
                if symptom in dsyms:
                    if value:
                        live_lh[d] *= 0.80
                    else:
                        live_lh[d] *= 0.20
                else:
                    if value:
                        live_lh[d] *= 0.15
                    else:
                        live_lh[d] *= 0.85
                        
        # 8. Calculate posteriors
        posteriors = {}
        total = 0.0
        for d in priors:
            p_temp = temp_lh.get(d, 0.5)
            p_hr = hr_lh.get(d, 0.5)
            p_spo2 = spo2_lh.get(d, 0.5)
            p_bp = bp_lh.get(d, 0.5)
            p_cam = camera_lh.get(d, 1.0)
            p_live = live_lh.get(d, 1.0)
            
            posterior = priors[d] * p_temp * p_hr * p_spo2 * p_bp * p_cam * p_live
            posteriors[d] = posterior
            total += posterior
            
        # 9. Normalize to sum to 100%
        normalized_scores = []
        if total > 0:
            for d, val in posteriors.items():
                normalized_scores.append((d, round((val / total) * 100)))
        else:
            for d in priors:
                normalized_scores.append((d, round(100 / len(priors))))
                
        # Sort by score descending
        ranked = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
        return ranked[:3]
    
    def get_differentiating_symptom(self, disease_a, disease_b):
        """Find a symptom in ALL_SYMPTOMS that is in one disease but not the other, 
        and has not been answered yet."""
        syms_a = DISEASE_SYMPTOMS.get(disease_a, [])
        syms_b = DISEASE_SYMPTOMS.get(disease_b, [])
        
        differentiators = []
        for sym in ALL_SYMPTOMS:
            if sym in self.live_symptoms_responses:
                continue
            in_a = sym in syms_a
            in_b = sym in syms_b
            if in_a != in_b:
                differentiators.append(sym)
                
        if differentiators:
            return differentiators[0]
        return None

    def on_live_yes_clicked(self):
        if self.current_differentiating_symptom:
            self.live_symptoms_responses[self.current_differentiating_symptom] = True
            self.update_live_diagnostics()

    def on_live_no_clicked(self):
        if self.current_differentiating_symptom:
            self.live_symptoms_responses[self.current_differentiating_symptom] = False
            self.update_live_diagnostics()

    def handle_worker_data(self, vitals_data, red_eye_detected, rash_detected):
        self.red_eye_detected = red_eye_detected
        self.rash_detected = rash_detected
        
        symptoms = []
        if self.red_eye_detected:
            symptoms.append("Red Eye")
        if self.rash_detected:
            symptoms.append("Rash")

        if symptoms:
            self.camera_status.setText("📷 Camera Symptoms: " + ", ".join(symptoms))
        else:
            self.camera_status.setText("📷 Camera Symptoms: None")

        # Synchronize camera-detected symptoms with manual checklist checkboxes in real-time
        if "Red eyes (Conjunctivitis)" in self.symptom_checkboxes:
            self.symptom_checkboxes["Red eyes (Conjunctivitis)"].setChecked(self.red_eye_detected)
            
        if "Skin rash" in self.symptom_checkboxes:
            self.symptom_checkboxes["Skin rash"].setChecked(self.rash_detected)

        if vitals_data is not None:
            try:
                temp = vitals_data["temperature"]
                heart_rate = vitals_data["heart_rate"]
                spo2 = vitals_data["spo2"]
                bp_sys = vitals_data["bp_sys"]
                bp_dia = vitals_data["bp_dia"]
                travel_history = vitals_data.get("travel_history", "none")
                
                # If in Live Simulator Mode, update the vitals card inputs in real-time
                if self.live_mode_cb.isChecked():
                    self.temp_val_input.blockSignals(True)
                    self.hr_val_input.blockSignals(True)
                    self.spo2_val_input.blockSignals(True)
                    self.bp_sys_input.blockSignals(True)
                    self.bp_dia_input.blockSignals(True)

                    self.temp_val_input.setText(f"{temp:.1f}")
                    self.hr_val_input.setText(f"{heart_rate}")
                    self.spo2_val_input.setText(f"{spo2}")
                    self.bp_sys_input.setText(f"{bp_sys}")
                    self.bp_dia_input.setText(f"{bp_dia}")

                    self.temp_val_input.blockSignals(False)
                    self.hr_val_input.blockSignals(False)
                    self.spo2_val_input.blockSignals(False)
                    self.bp_sys_input.blockSignals(False)
                    self.bp_dia_input.blockSignals(False)

                # Update manual entry temperature automatically when updated from mobile
                if self.last_live_vitals is None or self.last_live_vitals[0] != temp:
                    self.temp_input.setText(f"{temp:.1f}")

                # Update travel dropdown programmatically ONLY when travel history changes from server
                if self.last_live_vitals is None or self.last_live_vitals[5] != travel_history:
                    self.travel_input.blockSignals(True)
                    idx = self.travel_input.findData(travel_history)
                    if idx != -1:
                        self.travel_input.setCurrentIndex(idx)
                    self.travel_input.blockSignals(False)

                # Reset responses if live vitals change (i.e. new simulation run)
                current_vitals = (temp, heart_rate, spo2, bp_sys, bp_dia, travel_history)
                if self.last_live_vitals is not None and self.last_live_vitals != current_vitals:
                    self.live_symptoms_responses.clear()
                    self.current_differentiating_symptom = None
                self.last_live_vitals = current_vitals

                self.update_live_diagnostics()

                temp_c = (temp - 32) * 5 / 9
                self.draw_thermometer(temp_c)

                if temp > 105 or spo2 < 90:
                    self.status_live.setText("🔴 Patient Status : Critical")
                    self.status_live.setStyleSheet("color:#ff2d55;font-size:12px;font-weight:bold;")
                elif temp > 103 or spo2 < 94 or heart_rate > 120:
                    self.status_live.setText("🟠 Patient Status : High Risk")
                    self.status_live.setStyleSheet("color:#fab387;font-size:12px;font-weight:bold;")
                elif temp > 100 or heart_rate > 100:
                    self.status_live.setText("🟡 Patient Status : Monitor")
                    self.status_live.setStyleSheet("color:#f9e2af;font-size:12px;font-weight:bold;")
                else:
                    self.status_live.setText("🟢 Patient Status : Stable")
                    self.status_live.setStyleSheet("color:#a6e3a1;font-size:12px;font-weight:bold;")

                print(f"Temp={temp}°F | HR={heart_rate} | SpO2={spo2}% | BP={bp_sys}/{bp_dia}")
            except Exception as e:
                print("Vitals handle error:", e)

    def update_live_diagnostics(self):
        if self.last_live_vitals is None:
            return
        temp, heart_rate, spo2, bp_sys, bp_dia, travel_history = self.last_live_vitals
        try:
            # Update disease ranking
            top3 = self.get_possible_diseases(
                temp,
                heart_rate,
                spo2,
                bp_sys,
                bp_dia
            )

            region = self.travel_input.currentData()
            text = "🦟 Possible Diseases:\n\n"
            for i, (disease, score) in enumerate(top3, start=1):
                text += f"{i}. {disease} ({score}%)\n"
            if region == "none":
                text += "\n⚠️ Highly unlikely without travel history."
            self.possible_disease_lbl.setText(text)

            top1_name, top1_score = top3[0]
            top2_name, top2_score = top3[1]
            difference = top1_score - top2_score

            print(f"Top1={top1_name}({top1_score}) Top2={top2_name}({top2_score}) Diff={difference}")
            
            # Check for overlapping diagnosis
            diff_sym = None
            if difference < 20:
                diff_sym = self.get_differentiating_symptom(top1_name, top2_name)

            if diff_sym:
                if region == "none":
                    self.question_lbl.setText(
                        f"⚠️ Highly unlikely without travel history.\n"
                        f"Overlap detected: {top1_name} vs {top2_name}.\n"
                        f"Do you have {diff_sym.lower()}?"
                    )
                else:
                    self.question_lbl.setText(
                        f"Overlap detected: {top1_name} vs {top2_name}.\n"
                        f"Do you have {diff_sym.lower()}?"
                    )
                self.question_lbl.show()
                self.yes_btn.show()
                self.no_btn.show()
                self.current_differentiating_symptom = diff_sym
            else:
                if region == "none":
                    self.question_lbl.setText(f"⚠️ Highly unlikely without travel history.\nLikely: {top1_name}")
                else:
                    self.question_lbl.setText(f"✅ Diagnosis likely: {top1_name}")
                self.question_lbl.show()
                self.yes_btn.hide()
                self.no_btn.hide()
                self.current_differentiating_symptom = None
        except Exception as e:
            print("update_live_diagnostics error:", e)

    def on_live_mode_changed(self, is_live):
        self.temp_val_input.setEnabled(not is_live)
        self.hr_val_input.setEnabled(not is_live)
        self.spo2_val_input.setEnabled(not is_live)
        self.bp_sys_input.setEnabled(not is_live)
        self.bp_dia_input.setEnabled(not is_live)
        
        if not is_live:
            self.run_manual_vitals_check()

    def run_manual_vitals_check(self):
        try:
            temp = float(self.temp_val_input.text()) if self.temp_val_input.text() else 98.6
        except ValueError:
            temp = 98.6

        try:
            heart_rate = int(self.hr_val_input.text()) if self.hr_val_input.text() else 80
        except ValueError:
            heart_rate = 80

        try:
            spo2 = int(self.spo2_val_input.text()) if self.spo2_val_input.text() else 98
        except ValueError:
            spo2 = 98

        try:
            bp_sys = int(self.bp_sys_input.text()) if self.bp_sys_input.text() else 120
        except ValueError:
            bp_sys = 120

        try:
            bp_dia = int(self.bp_dia_input.text()) if self.bp_dia_input.text() else 80
        except ValueError:
            bp_dia = 80

        try:
            region = self.travel_input.currentData()
            self.last_live_vitals = (temp, heart_rate, spo2, bp_sys, bp_dia, region)
            self.update_live_diagnostics()

            temp_c = (temp - 32) * 5 / 9
            self.draw_thermometer(temp_c)
            
            if temp > 105 or spo2 < 90:
                self.status_live.setText("🔴 Patient Status : Critical")
                self.status_live.setStyleSheet("color:#ff2d55;font-size:12px;font-weight:bold;")
            elif temp > 103 or spo2 < 94 or heart_rate > 120:
                self.status_live.setText("🟠 Patient Status : High Risk")
                self.status_live.setStyleSheet("color:#fab387;font-size:12px;font-weight:bold;")
            elif temp > 100 or heart_rate > 100:
                self.status_live.setText("🟡 Patient Status : Monitor")
                self.status_live.setStyleSheet("color:#f9e2af;font-size:12px;font-weight:bold;")
            else:
                self.status_live.setText("🟢 Patient Status : Stable")
                self.status_live.setStyleSheet("color:#a6e3a1;font-size:12px;font-weight:bold;")
        except Exception as e:
            print("Manual vitals check error:", e)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()
    # ── DISEASE SELECTED ──────────────────────────────────────────────────────
    def on_disease_selected(self, name):
        if name == "-- Select a Disease --":
            self.load_body_model()
            self.symptom_list.clear()
            self.affected_list.clear()
            self.mosq_value.setText("—")
            self.severity_value.setText("—")
            self.draw_thermometer(37.0)
            self._set_table_placeholder()
            self.draw_placeholder_charts()
            return

        data = self.diseases[name]
        self.mosq_value.setText(data["mosquito"])

        sev_colors = {"Mild":"#a6e3a1","Moderate":"#f9e2af",
                      "High":"#fab387","Critical":"#f38ba8","Chronic":"#cba6f7"}
        sev = data["severity"]
        self.severity_value.setText(sev)
        self.severity_value.setStyleSheet(
            f"color:{sev_colors.get(sev,'#cdd6f4')};font-size:12px;"
            f"font-weight:bold;background:transparent;")

        self.symptom_list.clear()
        for s in data["symptoms"]:
            self.symptom_list.addItem(f"  • {s}")

        self.affected_list.clear()
        for organ in data["organs"]:
            self.affected_list.addItem(f"  🔴 {organ.replace('_',' ').title()}")

        self.draw_thermometer(data["temperature"])
        self.load_body_model(affected=data["organs"])
        self.draw_disease_charts(name)

        if "temperature_stages" in data:
            self.populate_stage_table(data["temperature_stages"])
        else:
            self._set_table_placeholder()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DigitalTwinApp()
    window.show()
    sys.exit(app.exec())