import os, sys, json, base64, subprocess, tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FRONTEND    = os.path.join(BASE_DIR, "..", "frontend")
RESULT_JSON = os.path.join(FRONTEND, "result.json")

app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # 1. ?��?지 ?�??
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "?��?지 ?�음"}), 400

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".jpg",
        dir=BASE_DIR
    )
    file.save(tmp.name)
    tmp.close()

    # 2. main.py ?�행
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "main.py"),
         "--image", tmp.name,
         "--output_json", RESULT_JSON],
        cwd=BASE_DIR,
        capture_output=True, text=True, encoding="utf-8"
    )
    os.unlink(tmp.name)

    if result.returncode != 0:
        return jsonify({"error": result.stderr}), 500

    # 3. result.json 반환
    with open(RESULT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

if __name__ == "__main__":
    print("?�길 ?�버: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
