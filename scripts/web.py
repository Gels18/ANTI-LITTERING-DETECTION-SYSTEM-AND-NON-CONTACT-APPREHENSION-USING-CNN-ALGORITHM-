from flask import Flask, render_template, Response, request, redirect
import cv2
from ultralytics import YOLO
import yagmail
import time
import json
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)  

model = YOLO("C:/Users/WINDOWS 10/Desktop/Litter-Detect/models/best.pt")
classes = model.names
last_sent = 0
email_delay = 10

def load_config():
    with open("scripts/config.json") as f:
        return json.load(f)

def save_config(data):
    with open("scripts/config.json", "w") as f:
        json.dump(data, f)

def send_email(filepath):
    config = load_config()
    yag = yagmail.SMTP(config["sender_email"], config["sender_password"])
    yag.send(
        to=config["receiver_email"],
        subject="Litter Detected",
        contents="Litter was detected. See attached screenshot.",
        attachments=filepath
    )
    print(f"[INFO] Email sent: {filepath}")

def gen_frames():
    global last_sent
    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, save=False)
        litter_detected = False

        for result in results:
            if len(result.boxes) > 0:
                litter_detected = True
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    clid = int(cls)
                    Clname = classes[clid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, Clname, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        now = time.time()
        if litter_detected and (now - last_sent > email_delay):
            filename = f"static/evidence/litter_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            try:
                send_email(filename)
                last_sent = now
            except Exception as e:
                print("[ERROR] Failed to send email:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    config = load_config()
    return render_template('index.html', config=config)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_email', methods=['POST'])
def update_email():
    new_config = {
        "sender_email": request.form["sender"],
        "sender_password": request.form["password"],
        "receiver_email": request.form["receiver"]
    }
    save_config(new_config)
    return redirect('/')

if __name__ == '__main__':
    os.makedirs("static/evidence", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
