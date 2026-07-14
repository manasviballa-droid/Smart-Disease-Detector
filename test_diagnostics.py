import sys

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

class MockTravelInput:
    def __init__(self, val):
        self.val = val
    def currentData(self):
        return self.val

class TestBayesianDiagnostics:
    def __init__(self):
        self.red_eye_detected = False
        self.rash_detected = False
        self.live_symptoms_responses = {}
        self.travel_input = MockTravelInput("none")

    def get_possible_diseases(self, temp, hr, spo2, bp_sys, bp_dia):
        # Copy-paste the exact same method from main.py
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
            
        temp_lh = {
            "Malaria": 0.90 if temp >= 104.0 else (0.60 if temp >= 102.0 else (0.20 if temp >= 99.5 else 0.05)),
            "Dengue": 0.85 if temp >= 103.0 else (0.50 if temp >= 101.0 else (0.20 if temp >= 99.5 else 0.05)),
            "Chikungunya": 0.90 if temp >= 102.0 else (0.40 if temp >= 100.0 else (0.15 if temp >= 99.0 else 0.05)),
            "Zika": 0.80 if 99.0 <= temp <= 101.5 else (0.40 if 101.5 < temp <= 103.0 else (0.10 if temp > 103.0 else 0.15)),
            "Yellow Fever": 0.80 if temp >= 103.0 else (0.40 if temp >= 101.0 else (0.15 if temp >= 99.0 else 0.05)),
            "West Nile Fever": 0.75 if 99.5 <= temp <= 102.0 else (0.30 if 102.0 < temp <= 104.0 else (0.10 if temp > 104.0 else 0.20)),
            "Lymphatic Filariasis": 0.70 if temp < 100.5 else (0.30 if temp <= 102.0 else 0.05)
        }

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
            
        normalized_scores = []
        if total > 0:
            for d, val in posteriors.items():
                normalized_scores.append((d, round((val / total) * 100)))
        else:
            for d in priors:
                normalized_scores.append((d, round(100 / len(priors))))
                
        ranked = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
        return ranked[:3]

# Run scenarios
test_app = TestBayesianDiagnostics()

print("--- TEST CASE 1: Sub-Saharan Africa travel, Extreme Fever (104.5) ---")
test_app.travel_input = MockTravelInput("africa")
results = test_app.get_possible_diseases(temp=104.5, hr=120, spo2=98, bp_sys=120, bp_dia=80)
print("Priors/Travel History region:", test_app.travel_input.currentData())
print("Expected top disease: Malaria")
print("Top 3 results:", results)

print("\n--- TEST CASE 2: South/Southeast Asia travel, Low BP (85/55), Fever (102.5) ---")
test_app.travel_input = MockTravelInput("asia")
results = test_app.get_possible_diseases(temp=102.5, hr=115, spo2=98, bp_sys=85, bp_dia=55)
print("Priors/Travel History region:", test_app.travel_input.currentData())
print("Expected top disease: Dengue")
print("Top 3 results:", results)

print("\n--- TEST CASE 3: South/Central America travel, Low-grade Fever (100.5), Red Eye ---")
test_app.travel_input = MockTravelInput("americas")
test_app.red_eye_detected = True
results = test_app.get_possible_diseases(temp=100.5, hr=85, spo2=98, bp_sys=120, bp_dia=80)
print("Priors/Travel History region:", test_app.travel_input.currentData())
print("Expected top disease: Zika")
print("Top 3 results:", results)

print("\n--- TEST CASE 4: No Travel/Local, high fever (103.5) ---")
test_app.travel_input = MockTravelInput("none")
results = test_app.get_possible_diseases(temp=103.5, hr=95, spo2=98, bp_sys=120, bp_dia=80)
print("Priors/Travel History region:", test_app.travel_input.currentData())
print("Top 3 results:", results)
