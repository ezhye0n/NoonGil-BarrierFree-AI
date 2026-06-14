import os, sys, json, base64, subprocess, tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import io  # 오류 처리로 추가

# 오류 처리를 위한 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
    # 1. 이미지 저장
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "이미지 없음"}), 400

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".jpg",
        dir=BASE_DIR
    )
    file.save(tmp.name)
    tmp.close()

    # 2. main.py 실행 (encoding="utf-8"로 이모지/한글 인코딩 오류 방지)
    # stdout/stderr를 파일로 저장
    stdout_file = tmp.name + ".stdout"
    stderr_file = tmp.name + ".stderr"

    proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, "main.py"),
         "--image", tmp.name,
         "--output_json", RESULT_JSON],
        cwd=BASE_DIR,
        stdout=open(stdout_file, "wb"),
        stderr=open(stderr_file, "wb"),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    proc.wait()
    os.unlink(tmp.name)

    with open(stderr_file, encoding="utf-8", errors="replace") as f:
        stderr_text = f.read()
    os.unlink(stdout_file)
    os.unlink(stderr_file)

    if proc.returncode != 0:
        return jsonify({"error": stderr_text}), 500

    with open(RESULT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

    # 3. result.json 반환
    with open(RESULT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

if __name__ == "__main__":
    print("눈길 서버: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
