"""
Quick test to verify face blurring works in data collection
"""
import cv2
import mediapipe as mp

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("="*60)
print("Face Blur Test")
print("="*60)
print("This will show your camera feed with face detection.")
print("Faces will appear with blur to test privacy protection.")
print("\nPress 'Q' to quit")
print("="*60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_blurred = frame.copy()
    
    # Detect faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    faces_detected = 0
    if results.detections:
        for detection in results.detections:
            faces_detected += 1
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            # Apply blur
            if w > 0 and h > 0:
                face_region = frame_blurred[y:y+h, x:x+w]
                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame_blurred[y:y+h, x:x+w] = face_region
                
                # Draw rectangle on original for comparison
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add status text
    cv2.putText(frame, f'Original (Faces detected: {faces_detected})', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_blurred, f'Privacy Protected (Blurred)', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame_blurred, 'This is how images are saved', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show both versions side by side
    combined = cv2.hconcat([frame, frame_blurred])
    cv2.imshow('Face Blur Test - Press Q to quit', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
face_detection.close()

print("\nTest complete!")
print("If faces were blurred, the privacy feature is working correctly.")
