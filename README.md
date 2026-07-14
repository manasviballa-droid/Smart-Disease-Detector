# Digital Twin — Mosquito-borne Disease Visualizer & Diagnostic System

An interactive desktop application that renders a **3D human body model** and visualizes organ pathology and diagnostic probabilities for mosquito-borne diseases in real time.

This application includes a **live patient vitals simulator** (via Flask), **computer vision webcam symptoms detection** (HSV-based rash and channel-ratio conjunctivitis eye check), and a **clinically validated Bayesian probability engine** mapping travel history priors to WHO case definitions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green) ![PyVista](https://img.shields.io/badge/PyVista-3D-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red) ![Flask](https://img.shields.io/badge/Flask-Server-black)

---

## Key Features

1. **3D Interactive Body Model**: Glows affected organs in red using a PyVista 3D canvas with full mouse navigation (rotate, zoom, pan).
2. **Clinical Bayesian Diagnostic Engine**: Replaces heuristic scoring with a rigorous Bayesian framework using travel history baseline priors and clinical likelihood ratios:
   - **Regional Priors**: Africa (Malaria, Yellow Fever), Asia (Dengue, Chikungunya), Americas (Zika, Dengue).
   - **Clinical Vitals Likelihoods**: Analyzes extreme/moderate fevers, relative bradycardia (Faget's sign in Yellow Fever), critical hypoxemia (low $SpO_2$), and hypotension (shock risk in Dengue/Yellow Fever).
3. **Webcam Symptoms Detection & Checklist Auto-Sync**: 
   - Webcam processing runs in a dedicated background `QThread` to ensure lag-free GUI rendering.
   - Detects **Red Eyes (Conjunctivitis)** and **Skin Rash** using optimized OpenCV Haar cascades and HSV masking.
   - Automatically checks/unchecks the corresponding manual checkboxes in real time.
4. **Dual Mode Vitals Entry (Manual & Live Telemetry)**:
   - **Enter Vitals Card**: Allows entering Temperature, Heart Rate, SpO₂, and Blood Pressure directly in the desktop application.
   - **Live Simulator**: Toggle to connect to the Flask vitals simulation server for real-time mobile/browser remote telemetry inputs.
5. **Interactive Overlap Resolution**: Automatically prompts the user with Catppuccin-themed **Yes** and **No** buttons to answer differential questions when candidate diseases overlap ($<20\%$ confidence difference).

---

## Project Structure

```
digital-twin/
├── main.py               # Main application - PyQt6 UI, 3D Canvas, Bayesian Engine
├── server.py             # Flask Server hosting the simulation API
├── index.html            # Web Controller page for sending live vitals
├── test_diagnostics.py   # Test suite verifying Bayesian scenario outcomes
├── diseases.json         # Disease catalog data (severity, organs, symptoms)
└── README.md             # Project documentation
```

---

## Requirements & Dependencies
 
 - Python 3.10+
 - PyQt6
 - PyVista & PyVistaQt
 - Matplotlib
 - OpenCV (opencv-contrib-python)
 - Requests
 - Flask & Flask-CORS
 
 Install all required packages:
 
 ```bash
 pip install pyqt6 pyvista pyvistaqt matplotlib opencv-contrib-python requests flask flask-cors
 ```
 
 ---
 
 ## Setup & Running the Application
 
 You can run the application entirely on your desktop using manual vitals entry, or optionally connect it with the live telemetry server.
 
 ### Option A: Run PyQt Desktop App Only (With Manual Entry)
 
 Start the main visualizer:
 ```bash
 python main.py
 ```
 Go to the **Checker** tab, scroll down to the **Enter Vitals** card, and type the vitals directly.
 
 ### Option B: Run with Live Telemetry Server (Optional)
 
 Open two terminal windows:
 
 1. **Terminal 1: Start the Vitals Server**
    ```bash
    python server.py
    ```
    This runs the simulation API on port `5000`. Open `http://127.0.0.1:5000/` in any browser to transmit simulated vitals.
 2. **Terminal 2: Run the PyQt Desktop App**
    ```bash
    python main.py
    ```
    Check the **Live Simulator** checkbox in the desktop app's vitals card to start receiving the telemetry.
 
 ---
 
 ## How to Test & Application Workflow
 
 1. **Enter Vitals**:
    * Open the application, select the **Checker** module tab, scroll down to **Enter Vitals**, and type in the vitals (Temperature, HR, SpO₂, and BP).
 2. **Camera Symptoms Detection (Auto-Sync)**:
    * Ensure your webcam is connected. The webcam runs in a dedicated background thread to prevent lag.
    * Hold a highly saturated red element (like a red card or red phone screen) in front of your face/eyes.
    * The camera will detect red eyes or a skin rash, display the results in the camera status bar, and **automatically check the corresponding boxes** in the symptoms list above.
 3. **Refine & Final Diagnosis**:
    * Scroll back up to the symptoms list, select any additional symptoms manually, and click **Final Diagnosis**.
    * If multiple diseases share overlapping symptoms (e.g. Dengue and Chikungunya), the system will prompt you with differential **Yes** and **No** questions.
 4. **Probable Fever Diagnosis**:
    * Answer the differential questions. The system will determine the most probable fever diagnosis and display it in the results field.
 
 ---
 
 ## Controls
 
 | Action | Control |
 |---|---|
 | Rotate model | Left click + drag |
 | Zoom | Scroll wheel |
 | Pan | Right click + drag |
 | Reset Checker | Click "Clear All" button |
 
 ---
 
 ## Disclaimer
 This project is educational only and does not provide professional medical advice.
