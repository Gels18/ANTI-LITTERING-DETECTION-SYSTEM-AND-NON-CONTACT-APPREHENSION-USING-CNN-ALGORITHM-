import cv2
from ultralytics import YOLO
import yagmail
import time
import os

model = YOLO("C:/Users/Administrator/Desktop/Litter-Detect/BAGO.v12i.yolov8/runs/detect/train5/weights/best.pt")
classes = model.names

yag = yagmail.SMTP("litteringanti@gmail.com", "hfxb giaf dznj dzif")
receiver = "eplapade@ccc.edu.ph"
evidence = ("C:/Users/Administrator/Desktop/Litter-Detect/evidence")
#os.makedirs(evidence, exist_ok=True)
cap = cv2.VideoCapture("C:/Users/Administrator/Downloads/BAGO.v12i.yolov8/sample.mp4")
#cap = cv2.VideoCapture(0)
 
Senttm = 0
Emaildelay = 10  

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break

    results = model(frame, save=False)
    litterdetect = False

    for result in results:
        if len(result.boxes) > 0:
            litterdetect = True
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                clid = int(cls)
                Clname = classes[clid]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, Clname, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    Realtime = time.time()
    if litterdetect and (Realtime - Senttm > Emaildelay):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"litter_{timestamp}.jpg"
        filepath = os.path.join(evidence, filename)
        cv2.imwrite(filepath, frame)

        try:
            yag.send(
                to=receiver,
                subject="Litter Detected",
                contents="Litter was detected. See attached screenshot.",
                attachments=filepath
            )
            print(f"[INFO] Email sent: {filename}")
            Senttm = Realtime
        except Exception as e:
            print("ERROR !!!  Failed to send email:", e)

    cv2.imshow("Litter Detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:   
        break

cap.release()
cv2.destroyAllWindows()
