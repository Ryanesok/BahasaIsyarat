"""
Sensor Fusion Module
Combines camera-based hand tracking with MPU6050 accelerometer data
for improved gesture recognition accuracy.
"""

import numpy as np
import time
from collections import deque
from typing import Dict, Optional, Tuple


class ComplementaryFilter:
    """
    Complementary filter for MPU6050 accelerometer data.
    Reduces noise and smooths readings.
    """
    def __init__(self, alpha=0.98):
        """
        Args:
            alpha: Filter coefficient (0.95-0.99). Higher = trust gyro more.
        """
        self.alpha = alpha
        self.filtered_accel = {'x': 0, 'y': 0, 'z': 0}
        self.initialized = False
        
    def update(self, accel_x: float, accel_y: float, accel_z: float) -> Dict[str, float]:
        """
        Apply complementary filter to accelerometer data.
        
        Args:
            accel_x, accel_y, accel_z: Raw accelerometer values
            
        Returns:
            Filtered accelerometer values as dict
        """
        if not self.initialized:
            # Initialize with first reading
            self.filtered_accel = {'x': accel_x, 'y': accel_y, 'z': accel_z}
            self.initialized = True
        else:
            # Apply low-pass filter
            self.filtered_accel['x'] = self.alpha * self.filtered_accel['x'] + (1 - self.alpha) * accel_x
            self.filtered_accel['y'] = self.alpha * self.filtered_accel['y'] + (1 - self.alpha) * accel_y
            self.filtered_accel['z'] = self.alpha * self.filtered_accel['z'] + (1 - self.alpha) * accel_z
        
        return self.filtered_accel.copy()
    
    def reset(self):
        """Reset filter state"""
        self.initialized = False
        self.filtered_accel = {'x': 0, 'y': 0, 'z': 0}


class MPUGestureDetector:
    """
    Detects gestures from filtered MPU6050 data.
    """
    def __init__(self, tilt_threshold=7000, shake_threshold=20000):
        self.tilt_threshold = tilt_threshold
        self.shake_threshold = shake_threshold
        self.filter = ComplementaryFilter(alpha=0.85)
        self.last_gesture = None
        self.last_gesture_time = 0
        self.cooldown = 1.5  # seconds
        
    def detect_gesture(self, accel_data: Dict) -> Tuple[Optional[str], float]:
        """
        Detect gesture from MPU accelerometer data.
        
        Args:
            accel_data: Dict with 'AcX', 'AcY', 'AcZ' keys
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        # Apply filtering
        filtered = self.filter.update(
            accel_data.get('AcX', 0),
            accel_data.get('AcY', 0),
            accel_data.get('AcZ', 0)
        )
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_gesture_time < self.cooldown:
            return None, 0.0
        
        ax = filtered['x']
        ay = filtered['y']
        
        gesture = None
        confidence = 0.0
        
        # Detect tilt gestures
        if abs(ax) > self.tilt_threshold:
            if ax > 0:
                gesture = "TILT_RIGHT"
                confidence = min(abs(ax) / 16000, 1.0) * 100  # Normalize to 0-100%
            else:
                gesture = "TILT_LEFT"
                confidence = min(abs(ax) / 16000, 1.0) * 100
        
        elif abs(ay) > self.tilt_threshold:
            if ay > 0:
                gesture = "TILT_FORWARD"
                confidence = min(abs(ay) / 16000, 1.0) * 100
            else:
                gesture = "TILT_BACKWARD"
                confidence = min(abs(ay) / 16000, 1.0) * 100
        
        if gesture:
            self.last_gesture = gesture
            self.last_gesture_time = current_time
            
        return gesture, confidence


class SensorFusion:
    """
    Fuses camera-based hand tracking with MPU accelerometer data
    to produce higher confidence gesture recognition.
    """
    def __init__(self, camera_weight=0.5, mpu_weight=0.5):
        """
        Args:
            camera_weight: Weight for camera-based detection (0-1)
            mpu_weight: Weight for MPU-based detection (0-1)
        """
        self.camera_weight = camera_weight
        self.mpu_weight = mpu_weight
        self.mpu_detector = MPUGestureDetector()
        self.mpu_data_queue = deque(maxlen=10)  # Last 10 MPU readings
        self.last_mpu_update = 0
        self.mpu_timeout = 2.0  # seconds
        
    def update_mpu_data(self, mpu_data: Dict):
        """
        Update MPU data from MQTT.
        
        Args:
            mpu_data: Dict containing accelerometer data
        """
        self.mpu_data_queue.append({
            'data': mpu_data,
            'timestamp': time.time()
        })
        self.last_mpu_update = time.time()
    
    def is_mpu_active(self) -> bool:
        """Check if MPU data is recent"""
        return (time.time() - self.last_mpu_update) < self.mpu_timeout
    
    def fuse_gestures(self, camera_gesture: Optional[str], camera_confidence: float,
                      mpu_gesture: Optional[str] = None, mpu_confidence: float = 0.0) -> Tuple[str, float]:
        """
        Fuse camera and MPU gesture data.
        
        Args:
            camera_gesture: Gesture detected by camera
            camera_confidence: Confidence of camera detection (0-100)
            mpu_gesture: Gesture detected by MPU (optional)
            mpu_confidence: Confidence of MPU detection (0-100)
            
        Returns:
            Tuple of (final_gesture, final_confidence)
        """
        # If no MPU data, use camera only
        if not self.is_mpu_active() or mpu_gesture is None:
            return camera_gesture if camera_gesture else "IDLE", camera_confidence
        
        # If gestures agree, boost confidence
        if camera_gesture and mpu_gesture and self._gestures_compatible(camera_gesture, mpu_gesture):
            combined_confidence = (camera_confidence * self.camera_weight + 
                                 mpu_confidence * self.mpu_weight) * 1.2  # Boost factor
            combined_confidence = min(combined_confidence, 100.0)
            return camera_gesture, combined_confidence
        
        # If gestures conflict, use weighted average
        if camera_confidence > mpu_confidence:
            return camera_gesture, camera_confidence * self.camera_weight
        else:
            return mpu_gesture, mpu_confidence * self.mpu_weight
    
    def _gestures_compatible(self, camera_gesture: str, mpu_gesture: str) -> bool:
        """
        Check if camera and MPU gestures are compatible/similar.
        """
        # Define compatible gesture pairs
        compatible_pairs = [
            ("SWIPE_RIGHT", "TILT_RIGHT"),
            ("SWIPE_LEFT", "TILT_LEFT"),
            ("PUSH_FORWARD", "TILT_FORWARD"),
            ("PULL_BACK", "TILT_BACKWARD"),
        ]
        
        pair = (camera_gesture, mpu_gesture)
        reverse_pair = (mpu_gesture, camera_gesture)
        
        return pair in compatible_pairs or reverse_pair in compatible_pairs
    
    def get_mpu_gesture(self) -> Tuple[Optional[str], float]:
        """
        Get current MPU gesture detection.
        
        Returns:
            Tuple of (gesture, confidence)
        """
        if not self.mpu_data_queue:
            return None, 0.0
        
        latest_data = self.mpu_data_queue[-1]['data']
        return self.mpu_detector.detect_gesture(latest_data)
    
    def reset(self):
        """Reset fusion state"""
        self.mpu_data_queue.clear()
        self.mpu_detector.filter.reset()
        self.last_mpu_update = 0


class GestureDecisionEngine:
    """
    Final decision layer for gesture recognition.
    Applies temporal smoothing and confidence thresholds.
    """
    def __init__(self, min_confidence=60.0, stability_frames=3):
        self.min_confidence = min_confidence
        self.stability_frames = stability_frames
        self.gesture_history = deque(maxlen=stability_frames)
        self.last_confirmed_gesture = None
        
    def decide(self, gesture: str, confidence: float) -> Tuple[Optional[str], float, str]:
        """
        Make final gesture decision with temporal smoothing.
        
        Args:
            gesture: Proposed gesture
            confidence: Confidence level (0-100)
            
        Returns:
            Tuple of (confirmed_gesture, confidence, state)
            state can be: "LISTENING", "STABLE", "COOLDOWN", "IDLE"
        """
        # Check minimum confidence threshold
        if confidence < self.min_confidence:
            self.gesture_history.clear()
            return None, confidence, "IDLE"
        
        # Add to history
        self.gesture_history.append(gesture)
        
        # Check if gesture is stable across frames
        if len(self.gesture_history) >= self.stability_frames:
            if all(g == gesture for g in self.gesture_history):
                # Stable gesture detected
                self.last_confirmed_gesture = gesture
                self.gesture_history.clear()
                return gesture, confidence, "STABLE"
            else:
                return None, confidence, "LISTENING"
        else:
            return None, confidence, "LISTENING"
    
    def reset(self):
        """Reset decision state"""
        self.gesture_history.clear()
        self.last_confirmed_gesture = None
