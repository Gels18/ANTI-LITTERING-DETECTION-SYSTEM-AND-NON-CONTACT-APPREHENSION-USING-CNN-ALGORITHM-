import cv2 
import mediapipe as mp
from ultralytics import YOLO


mpfd = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
model = YOLO("C:/Users/Administrator/Desktop/Litter-Detect/models/best.pt")
classes = model.names 
cap = cv2.VideoCapture("C:/Users/Administrator/Desktop/Litter-Detect/BAGO.v12i.yolov8/bote.mp4")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('bote.mp4', fourcc, fps, (width, height)) #where the output of the sample vid will be saved


with mpfd.FaceDetection(min_detection_confidence=0.5) as facedt:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rsltface = facedt.process(image) 
        rsltlit = model(frame, save=False) 

                                                            # face bounding boxes
        if rsltface.detections:
            for detection in rsltface.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                                    #litter detection boxes
        for result in rsltlit:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                clid = int(cls)
                Clname = classes[clid]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, Clname, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  
        cv2.imshow('Litter', frame)
        out.write(frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
