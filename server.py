from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

latest_data = {
    "temperature": 98.6,
    "heart_rate": 72,
    "spo2": 98,
    "bp_sys": 120,
    "bp_dia": 80,
    "travel_history": "none"
}

@app.route("/update", methods=["POST"])
def update():
    global latest_data
    data = request.get_json()
    if data:
        latest_data["temperature"] = float(data.get("temperature", latest_data["temperature"]))
        latest_data["heart_rate"] = int(data.get("heart_rate", latest_data["heart_rate"]))
        latest_data["spo2"] = int(data.get("spo2", latest_data["spo2"]))
        latest_data["bp_sys"] = int(data.get("bp_sys", latest_data["bp_sys"]))
        latest_data["bp_dia"] = int(data.get("bp_dia", latest_data["bp_dia"]))
        latest_data["travel_history"] = str(data.get("travel_history", latest_data["travel_history"]))
    return jsonify({"status": "success", "data": latest_data})

@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(latest_data)

@app.route("/")
def home():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)