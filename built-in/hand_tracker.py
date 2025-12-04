import os
import sys

# CRITICAL: Redirect stderr at file descriptor level
if sys.platform == 'win32' and not hasattr(sys, '_stderr_redirected'):
    try:
        stderr_fd = sys.stderr.fileno()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        sys.stderr = open(os.devnull, 'w')
        sys._stderr_redirected = True
    except:
        pass

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
import numpy as np
import mediapipe as mp

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe Hand Detection with optimized settings for glove detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # More stable settings with smoothing to reduce flickering
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,           # Focus on one hand only
            min_detection_confidence=0.7,  # Higher - more stable initial detection
            min_tracking_confidence=0.7,   # Higher - smoother tracking
            model_complexity=1         # Complex model for better accuracy
        )
        
        self.last_position = "CENTRE"
        self.detection_history = []
        
        print("[INFO] HandTracker initialized with glove-optimized MediaPipe")
    
    def track(self, frame):
        """Track hand using MediaPipe 21-point detection
        Returns: processed_frame, gesture, confidence
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        gesture = "NOT DETECTED"
        confidence = 0.0
        hand_info = {}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_handedness = results.multi_handedness[0]
            
            # Draw hand landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get hand center position (palm center - landmark 9)
            h, w, _ = frame.shape
            palm_center = hand_landmarks.landmark[9]
            cX = int(palm_center.x * w)
            cY = int(palm_center.y * h)
            
            # Determine position
            if cX < w / 3:
                gesture = "LEFT"
            elif cX > 2 * w / 3:
                gesture = "RIGHT"
            else:
                gesture = "CENTRE"
            
            # Calculate confidence based on hand detection score
            confidence = hand_handedness.classification[0].score * 100
            
            # Store hand info for developer console
            hand_info = {
                'landmarks_count': len(hand_landmarks.landmark),
                'handedness': hand_handedness.classification[0].label,
                'position': (cX, cY),
                'gesture': gesture,
                'confidence': confidence,
                'landmarks': hand_landmarks  # Pass full hand_landmarks object
            }
            
            # Update history for smoothing
            self.detection_history.append(gesture)
            if len(self.detection_history) > 5:
                self.detection_history.pop(0)
            
            # Use most common gesture from history
            if len(self.detection_history) >= 3:
                gesture = max(set(self.detection_history), key=self.detection_history.count)
            
            self.last_position = gesture
            
            # Draw position indicator
            color = (0, 255, 255) if gesture == "LEFT" else (255, 0, 255) if gesture == "RIGHT" else (0, 255, 0)
            cv2.circle(frame, (cX, cY), 15, color, -1)
            cv2.circle(frame, (cX, cY), 18, (255, 255, 255), 2)
            cv2.putText(frame, f"{gesture} ({confidence:.0f}%)", (cX - 80, cY - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw wrist to palm line for visualization
            wrist = hand_landmarks.landmark[0]
            wX, wY = int(wrist.x * w), int(wrist.y * h)
            cv2.line(frame, (wX, wY), (cX, cY), color, 3)
        
        return frame, gesture, confidence, hand_info
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
            print("[INFO] HandTracker closed")
