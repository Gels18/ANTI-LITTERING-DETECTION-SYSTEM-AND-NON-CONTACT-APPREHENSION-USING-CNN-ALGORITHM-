import cv2  
import mediapipe as mp  

mp_face_detection = mp.solutions.face_detection  
mp_drawing = mp.solutions.drawing_utils  
cap = cv2.VideoCapture(0) #kaya 0 kasi default webcam camera ang gamit ko

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break  

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        results = face_detection.process(image)  

        if results.detections:  # Check kung may nadetect na mukha
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)  # I-draw ang bounding box sa mukha

        cv2.imshow('Face Detection', frame) 

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key para mag-exit
            break

cap.release() 
cv2.waitKey(1)  # Lagyan ng delay para maiwasang mag crash
cv2.destroyAllWindows()  

#kaya po mediapipe and ginamit since mas mabilis po sya kesa sa tensorflow 