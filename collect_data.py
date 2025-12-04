"""
Professional Sign Language Data Collection System
Supports multiple data types with intelligent classification:
- STATIC: Single characters (A-Z) and numbers (0-9) - saved as images
- DYNAMIC: Words, sentences, phrases - saved as video sequences
- PRIVACY: Automatic face blurring for all captured data
"""

import os
import sys

# CRITICAL: Redirect stderr at file descriptor level
if sys.platform == 'win32':
    stderr_fd = sys.stderr.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    sys.stderr = open(os.devnull, 'w')

# Suppress ALL warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
from datetime import datetime
import mediapipe as mp

# Professional data structure
BASE_DIR = './sign-language-detector-python'
DATA_STRUCTURE = {
    'static': {
        'alphabet': os.path.join(BASE_DIR, 'data', 'static', 'alphabet'),  # A-Z
        'numbers': os.path.join(BASE_DIR, 'data', 'static', 'numbers')      # 0-9
    },
    'dynamic': {
        'words': os.path.join(BASE_DIR, 'data', 'dynamic', 'words'),        # Single words
        'phrases': os.path.join(BASE_DIR, 'data', 'dynamic', 'phrases'),    # Short phrases
        'sentences': os.path.join(BASE_DIR, 'data', 'dynamic', 'sentences') # Full sentences
    }
}

# Create directory structure
for category in DATA_STRUCTURE.values():
    for path in category.values():
        os.makedirs(path, exist_ok=True)

# Display professional interface
print("="*70)
print("PROFESSIONAL SIGN LANGUAGE DATA COLLECTION SYSTEM")
print("="*70)
print("\n📂 Data Classification System:")
print("   STATIC (Images):")
print("   ├── Alphabet: A-Z single characters")
print("   └── Numbers: 0-9 digits")
print("\n   DYNAMIC (Videos):")
print("   ├── Words: Single word gestures")
print("   ├── Phrases: 2-5 word combinations")
print("   └── Sentences: Complex multi-word expressions")
print("\n" + "="*70)

# Get collection type
print("\nSelect data type to collect:")
print("1. Static - Alphabet (A-Z)")
print("2. Static - Numbers (0-9)")
print("3. Dynamic - Words (video)")
print("4. Dynamic - Phrases (video)")
print("5. Dynamic - Sentences (video)")
print()

collection_type = input("Enter choice (1-5): ").strip()

if collection_type == '1':
    data_type = 'static'
    category = 'alphabet'
    print("\n📝 Collecting: ALPHABET characters")
    items_input = input("Enter letters (e.g., ABC or A,B,C): ").strip().upper()
    items = [l.strip() for l in items_input.replace(',', '').replace(' ', '')]
    dataset_size = 150
    is_video = False
    
elif collection_type == '2':
    data_type = 'static'
    category = 'numbers'
    print("\n📝 Collecting: NUMBER digits")
    items_input = input("Enter numbers (e.g., 123 or 1,2,3): ").strip()
    items = [n.strip() for n in items_input.replace(',', '').replace(' ', '')]
    dataset_size = 150
    is_video = False
    
elif collection_type == '3':
    data_type = 'dynamic'
    category = 'words'
    print("\n🎬 Collecting: WORDS (video)")
    items_input = input("Enter words separated by comma (e.g., hello,goodbye,thanks): ").strip().lower()
    items = [w.strip() for w in items_input.split(',')]
    dataset_size = 20  # Fewer videos needed
    is_video = True
    
elif collection_type == '4':
    data_type = 'dynamic'
    category = 'phrases'
    print("\n🎬 Collecting: PHRASES (video)")
    items_input = input("Enter phrases separated by | (e.g., good morning|thank you|how are you): ").strip().lower()
    items = [p.strip() for p in items_input.split('|')]
    dataset_size = 15
    is_video = True
    
elif collection_type == '5':
    data_type = 'dynamic'
    category = 'sentences'
    print("\n🎬 Collecting: SENTENCES (video)")
    items_input = input("Enter sentences separated by | : ").strip().lower()
    items = [s.strip() for s in items_input.split('|')]
    dataset_size = 10
    is_video = True
    
else:
    print("Invalid choice. Exiting.")
    exit()

if not items:
    print("No items specified. Exiting.")
    exit()

save_path = DATA_STRUCTURE[data_type][category]
print(f"\n✓ Will collect {dataset_size} {'videos' if is_video else 'images'} for: {', '.join(items)}")
print(f"✓ Save location: {save_path}")

# Initialize face detection for privacy (blur faces)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)
print("\n[INFO] Face detection enabled - faces will be automatically blurred")

# Initialize hand detection for real-time feedback
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
print("[INFO] Hand detection enabled - real-time feedback active")

# Version validation
print("\n" + "="*70)
print("✓ USING FIXED VERSION - Clean image save (no landmarks)")
print("="*70)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

# Set larger frame size for better visibility
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Instructions based on type
if is_video:
    print("\n🎬 VIDEO RECORDING INSTRUCTIONS:")
    print("1. Press 'Q' to START recording")
    print("2. Perform the sign language gesture")
    print("3. Press 'Q' again to STOP recording")
    print("4. Each recording will be 3-5 seconds")
else:
    print("\n📸 IMAGE CAPTURE INSTRUCTIONS:")
    print("1. Use GOOD LIGHTING")
    print("2. Show gesture from MULTIPLE ANGLES")
    print("3. Show gesture at DIFFERENT DISTANCES")
    print("4. HOLD STEADY during capture")

# Collect data for each item
for item in items:
    item_dir = os.path.join(save_path, item)
    os.makedirs(item_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Collecting: {item.upper()} ({'VIDEO' if is_video else 'IMAGES'})")
    print(f"{'='*70}")
    
    if is_video:
        # VIDEO RECORDING MODE
        for video_num in range(dataset_size):
            print(f"\nVideo {video_num + 1}/{dataset_size}")
            print("Position yourself and press 'Q' to START recording...")
            
            # Wait for start signal
            recording = False
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                if not recording:
                    cv2.putText(frame, f'{item.upper()}', (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f'Video {video_num + 1}/{dataset_size}', (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    cv2.putText(frame, 'Press Q to START', (50, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                else:
                    # Recording in progress
                    cv2.putText(frame, 'RECORDING...', (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(frame, f'Frame: {len(frames)}', (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(frame, 'Press Q to STOP', (50, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    # Blur faces in recording
                    frame_blurred = frame.copy()
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            x = max(0, int(bboxC.xmin * iw))
                            y = max(0, int(bboxC.ymin * ih))
                            w = min(int(bboxC.width * iw), iw - x)
                            h = min(int(bboxC.height * ih), ih - y)
                            
                            if w > 0 and h > 0:
                                face_region = frame_blurred[y:y+h, x:x+w]
                                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                                frame_blurred[y:y+h, x:x+w] = face_region
                    
                    frames.append(frame_blurred)
                
                cv2.imshow('Data Collection', frame)
                key = cv2.waitKey(25)
                
                if key == ord('q') or key == ord('Q'):
                    if not recording:
                        recording = True
                        print("Recording started...")
                    else:
                        print(f"Recording stopped. Captured {len(frames)} frames.")
                        break
            
            # Save video
            if len(frames) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(item_dir, f'{timestamp}_v{video_num}.mp4')
                
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                for frame in frames:
                    out.write(frame)
                
                out.release()
                print(f"✓ Video saved: {video_path}")
        
    else:
        # IMAGE CAPTURE MODE (existing code)
        print("Position your hand and press 'Q' when ready...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect hand for real-time feedback
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands_detector.process(rgb_frame)
            
            # Draw hand landmarks if detected
            hand_detected = False
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
            
            # Display with feedback
            cv2.putText(frame, f'{item.upper()}', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
            cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # Hand detection status
            if hand_detected:
                cv2.putText(frame, 'HAND DETECTED', (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'NO HAND - Position closer!', (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Instructions box
            cv2.putText(frame, 'TIPS FOR QUALITY:', (50, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, '1. Good lighting required', (50, 320),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, '2. Hand fills 30-50% of frame', (50, 350),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, '3. Clear gesture visibility', (50, 380),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, '4. Vary position slightly', (50, 410),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(25) == ord('q') or cv2.waitKey(25) == ord('Q'):
                break
        
        # Capture images
        print(f"Capturing {dataset_size} images...")
        counter = 0
        valid_captures = 0
        skipped_no_hand = 0
        
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Check if hand is detected BEFORE saving
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands_detector.process(rgb_frame)
            
            hand_detected = False
            if hand_results.multi_hand_landmarks:
                hand_detected = True
            
            # Create clean copy for saving BEFORE drawing landmarks
            frame_to_save = frame.copy()
            
            # NOW draw landmarks on display frame only (not saved)
            if hand_detected:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
            
            # Blur faces on the CLEAN copy (not the one with drawn landmarks)
            frame_blurred = frame_to_save.copy()
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    # Calculate face bounding box
                    face_x = max(0, int(bboxC.xmin * iw))
                    face_y = max(0, int(bboxC.ymin * ih))
                    face_w = min(int(bboxC.width * iw), iw - face_x)
                    face_h = min(int(bboxC.height * ih), ih - face_y)
                    
                    if face_w > 0 and face_h > 0:
                        # Check if hand landmarks are inside face region
                        hand_in_face = False
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                for landmark in hand_landmarks.landmark:
                                    lm_x = int(landmark.x * iw)
                                    lm_y = int(landmark.y * ih)
                                    
                                    # Check if landmark is inside face box
                                    if (face_x <= lm_x <= face_x + face_w and 
                                        face_y <= lm_y <= face_y + face_h):
                                        hand_in_face = True
                                        break
                                if hand_in_face:
                                    break
                        
                        # Only blur face if hand is NOT overlapping
                        if not hand_in_face:
                            face_region = frame_blurred[face_y:face_y+face_h, face_x:face_x+face_w]
                            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                            frame_blurred[face_y:face_y+face_h, face_x:face_x+face_w] = face_region
            
            # Display info
            cv2.putText(frame, f'{item.upper()}', (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
            cv2.putText(frame, f'Progress: {valid_captures}/{dataset_size}', (50, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            if hand_detected:
                cv2.putText(frame, 'CAPTURING - Hand OK', (50, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'SKIPPING - No hand detected!', (50, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, 'Move hand CLOSER to camera', (50, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            if skipped_no_hand > 0:
                cv2.putText(frame, f'Skipped (no hand): {skipped_no_hand}', (50, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(25)
            
            # Only save if hand is detected
            if hand_detected:
                img_path = os.path.join(item_dir, f'{counter}.jpg')
                cv2.imwrite(img_path, frame_blurred)
                valid_captures += 1
                counter += 1
            else:
                skipped_no_hand += 1
                # Still increment counter to avoid infinite loop
                counter += 1
        
        print(f"✓ Completed {valid_captures}/{dataset_size} images for '{item}'")
        if skipped_no_hand > 0:
            print(f"⚠ Skipped {skipped_no_hand} frames (no hand detected)")
            print("  TIP: Ensure hand is clearly visible and well-lit for next collection")

cap.release()
cv2.destroyAllWindows()
face_detection.close()
hands_detector.close()

print("\n" + "="*70)
print("✓ DATA COLLECTION COMPLETE!")
print(f"✓ Saved to: {save_path}")
print("[INFO] All data has faces blurred for privacy protection")
print("\n📊 Data Structure:")
print(f"   {data_type.upper()} > {category.upper()} > {', '.join(items)}")
print("\nNext steps:")
if not is_video:
    print("1. Run: python train_model.py")
    print("2. Then run your desktop application")
else:
    print("1. Process videos for training (create video processing script)")
    print("2. Train dynamic gesture recognition model")
print("="*70)
