import cv2 
import mediapipe as mp
from ultralytics import YOLO

mpfd = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
model = YOLO("C:/Users/Administrator/Desktop/Litter-Detect/BAGO.v12i.yolov8/runs/detect/train5/weights/best.pt")  # Path sa trained model
classes = model.names  # class names na ginamit sa trained dataset
cap = cv2.VideoCapture("C:/Users/Administrator/Downloads/BAGO.v12i.yolov8/vid6.mp4")

with mpfd.FaceDetection(min_detection_confidence=0.5) as facedt:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rsltface = facedt.process(image)  # Face detection
        rsltlit = model(frame, save=False)

                       
        if rsltface.detections:                #  bounding boxes for face 
            for detection in rsltface.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Add Face text

      
        for result in rsltlit: # bounding boxes for litter 
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                clid = int(cls)  # Convert class index to integer
                Clname = classes[clid]  # Get class name from index
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, Clname, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Litter & Face Detection', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows() 
