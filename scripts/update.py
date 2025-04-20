import cv2
import mediapipe as mp
from ultralytics import YOLO
import yagmail   
import time      
import os      



 
mpfd = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
model = YOLO("C:/Users/Administrator/Desktop/Litter-Detect/BAGO.v12i.yolov8/runs/detect/train5/weights/best.pt")
classes = model.names
yag = yagmail.SMTP("litteringanti@gmail.com", "Tsunayoshi081801")  # ✅ NEW
receiver = "eplapade@ccc.edu.ph"

cap = cv2.VideoCapture(0)

with mpfd.FaceDetection(min_detection_confidence=0.5) as facedt:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
        rsltface = facedt.process(image)
        rsltlit = model(frame, save=False)


        face_detected = False
        litter_detected = False

     
        if rsltface.detections:
            face_detected = True
            for detection in rsltface.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

   
        for result in rsltlit:
            if len(result.boxes) > 0:
                litter_detected = True
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                clid = int(cls)
                Clname = classes[clid]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, Clname, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      
        if face_detected and litter_detected:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"evidence_{timestamp}.jpg"
            filepath = os.path.join("C:/Users/Administrator/Desktop/Litter-Detect/evidence", filename)
            cv2.imwrite(filepath, frame)

        
            try:
                yag.send(
                    to=receiver,
                    subject="Littering Incident Detected 🚨",
                    contents="Face and litter detected. See attached image.",
                    attachments=filepath
                )
                print("Email sent successfully.")  # ✅ NEW
                time.sleep(10)  # ✅ Avoid spamming email every frame
            except Exception as e:
                print("Email failed:", e)

        cv2.imshow('Litter & Face Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
