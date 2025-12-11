"""
Sign Language Alphabet Classifier
Uses MediaPipe hand landmarks with RandomForestClassifier for alphabet recognition
Based on: https://github.com/computervisioneng/sign-language-detector-python
"""

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
from pathlib import Path
import mediapipe as mp
import pickle
import time
from sklearn.ensemble import RandomForestClassifier

# Import path configuration for proper .exe support
try:
    from path_config import PROJECT_ROOT, MODEL_FILE, DATA_ROOT
except ImportError:
    # Fallback if path_config not available
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_FILE = PROJECT_ROOT / "built-in" / "dataset" / "model.p"
    DATA_ROOT = PROJECT_ROOT / "sign-language-detector-python" / "data"

class AlphabetClassifier:
    def __init__(self, model_path=None, data_path=None):
        """
        Initialize alphabet classifier with pre-trained model or prepare for training
        
        Args:
            model_path: Path to trained model pickle file (uses path_config if None)
            data_path: Path to training data images (uses path_config if None)
        """
        # Use path_config paths for .exe compatibility
        self.model_path = Path(MODEL_FILE) if model_path is None else Path(model_path)
        self.data_path = Path(DATA_ROOT) if data_path is None else Path(data_path)
        
        print(f"[INFO] AlphabetClassifier model path: {self.model_path}")
        print(f"[INFO] Data path: {self.data_path}")
        
        # MediaPipe hands setup for landmark extraction
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Dynamic mode for video
            max_num_hands=1,
            min_detection_confidence=0.5,  # Higher for more stable detection
            min_tracking_confidence=0.5
        )
        
        # Model and labels
        self.model = None
        self.labels_dict = {}
        self.static_classes = []
        self.dynamic_classes = []
        
        # Load model if exists
        if self.model_path.exists():
            self._load_model()
        else:
            print("[WARN] No trained model found. Use train_model() to create one.")
        
        # Recognition stability parameters
        self.stability_frames = 15
        self.gesture_history = []
        self.last_recognized = None
        self.last_recognition_time = 0
        self.cooldown_period = 2.5
        
        # Dynamic gesture buffer (for word recognition)
        self.dynamic_gesture_buffer = []
        self.dynamic_buffer_max_frames = 45  # ~1.5 seconds at 30fps
        self.dynamic_recognition_threshold = 0.7  # 70% confidence
        
        print("[INFO] AlphabetClassifier initialized")
    
    def _load_model(self):
        """Load trained model from pickle file"""
        try:
            with open(self.model_path, 'rb') as f:
                model_dict = pickle.load(f)
                self.model = model_dict['model']
                # Create labels dictionary if not in file
                if 'labels_dict' in model_dict:
                    self.labels_dict = model_dict['labels_dict']
                else:
                    # Default labels for demo (customize based on your training)
                    self.labels_dict = {0: 'A', 1: 'B', 2: 'L'}
                
                # Load static/dynamic class separation
                self.static_classes = model_dict.get('static_classes', [])
                self.dynamic_classes = model_dict.get('dynamic_classes', [])
                
                if self.dynamic_classes:
                    print(f"[INFO] Dynamic gestures loaded: {len(self.dynamic_classes)} words/phrases")
                    print(f"  - Examples: {', '.join(self.dynamic_classes[:5])}")
                
                # AUTO-DETECT: Check if model expects 2D (42 features) or 3D (63 features)
                expected_features = getattr(self.model, 'n_features_in_', 63)
                self.use_3d_features = (expected_features == 63)
                
                if not self.use_3d_features:
                    print(f"[WARN] Model trained with OLD 2D features ({expected_features} dims)")
                    print("[WARN] Running in compatibility mode. For better accuracy, retrain:")
                    print("       1. python collect_data.py  (150+ images per letter)")
                    print("       2. python train_model.py   (creates new 3D model)")
                else:
                    print(f"[INFO] Model using ENHANCED 3D features ({expected_features} dims)")
                
                print(f"[INFO] Model loaded successfully with {len(self.labels_dict)} classes")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
            self.use_3d_features = True  # Default to 3D for new models
    
    def extract_hand_features(self, hand_landmarks):
        """
        Extract ENHANCED normalized features from hand landmarks
        Uses improved normalization with scaling for better discrimination
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object (has .landmark attribute) OR the landmark list directly
            
        Returns:
            numpy array of normalized features or None
        """
        try:
            data_aux = []
            x_ = []
            y_ = []
            z_ = []
            
            # Handle both hand_landmarks object and direct landmark list
            if hasattr(hand_landmarks, 'landmark'):
                landmarks_list = hand_landmarks.landmark
            elif hasattr(hand_landmarks, '__iter__'):
                # It's already a list/RepeatedCompositeContainer of landmarks
                landmarks_list = hand_landmarks
            else:
                raise ValueError("Invalid hand_landmarks format")
            
            # Collect all x, y, z coordinates
            for i in range(len(landmarks_list)):
                x_.append(landmarks_list[i].x)
                y_.append(landmarks_list[i].y)
                z_.append(landmarks_list[i].z)
            
            # Calculate bounding box dimensions for scaling
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            min_z, max_z = min(z_), max(z_)
            
            # Calculate ranges (avoid division by zero)
            range_x = max_x - min_x if max_x - min_x > 0.001 else 0.001
            range_y = max_y - min_y if max_y - min_y > 0.001 else 0.001
            range_z = max_z - min_z if max_z - min_z > 0.001 else 0.001
            
            # Build feature vector based on model compatibility mode
            if getattr(self, 'use_3d_features', True):
                # NEW: 3D features (X, Y, Z) = 63 features for better accuracy
                for i in range(len(landmarks_list)):
                    data_aux.append((x_[i] - min_x) / range_x)  # Normalized X
                    data_aux.append((y_[i] - min_y) / range_y)  # Normalized Y
                    data_aux.append((z_[i] - min_z) / range_z)  # Normalized Z (depth)
            else:
                # OLD: 2D features (X, Y only) = 42 features for backward compatibility
                for i in range(len(landmarks_list)):
                    data_aux.append((x_[i] - min_x) / range_x)
                    data_aux.append((y_[i] - min_y) / range_y)
            
            return np.asarray(data_aux)
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None

    
    def create_dataset_from_images(self, output_file="data.pickle"):
        """
        Create training dataset from collected images
        Should be run after collecting images with collect_imgs.py
        
        Args:
            output_file: Output pickle file path
        """
        script_dir = Path(__file__).parent
        output_path = script_dir / output_file
        
        if not self.data_path.exists():
            print(f"[ERROR] Data path not found: {self.data_path}")
            print("[INFO] Run collect_imgs.py first to collect training data")
            return
        
        print("[INFO] Creating dataset from images...")
        data = []
        labels = []
        
        for dir_ in os.listdir(self.data_path):
            dir_path = self.data_path / dir_
            if not dir_path.is_dir():
                continue
                
            print(f"[INFO] Processing class: {dir_}")
            for img_name in os.listdir(dir_path):
                img_path = dir_path / img_name
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        features = self.extract_hand_features(hand_landmarks)
                        data.append(features)
                        labels.append(dir_)
        
        # Save dataset
        with open(output_path, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        
        print(f"[INFO] Dataset created: {len(data)} samples")
        print(f"[INFO] Saved to: {output_path}")
    
    def train_model(self, dataset_file="data.pickle", output_model="model.p"):
        """
        Train RandomForestClassifier on the dataset
        
        Args:
            dataset_file: Input pickle file with training data
            output_model: Output model file
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        script_dir = Path(__file__).parent
        dataset_path = script_dir / dataset_file
        model_output_path = script_dir / output_model
        
        if not dataset_path.exists():
            print(f"[ERROR] Dataset file not found: {dataset_path}")
            print("[INFO] Run create_dataset_from_images() first")
            return
        
        print("[INFO] Loading dataset...")
        with open(dataset_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])
        
        print(f"[INFO] Dataset: {len(data)} samples")
        
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )
        
        # Train model with optimized parameters for better accuracy
        print("[INFO] Training RandomForestClassifier with optimized parameters...")
        model = RandomForestClassifier(
            n_estimators=200,        # More trees for better accuracy
            max_depth=25,            # Deeper trees
            min_samples_split=3,     # Require more samples to split
            min_samples_leaf=1,      # Allow fine-grained decisions
            random_state=42,         # Reproducible results
            n_jobs=-1                # Use all CPU cores
        )
        model.fit(x_train, y_train)
        
        # Test accuracy
        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)
        print(f"[INFO] Model accuracy: {score * 100:.2f}%")
        
        # Create labels dictionary
        unique_labels = sorted(set(labels))
        labels_dict = {i: label.upper() for i, label in enumerate(unique_labels)}
        
        # Save model
        with open(model_output_path, 'wb') as f:
            pickle.dump({'model': model, 'labels_dict': labels_dict}, f)
        
        print(f"[INFO] Model saved to: {model_output_path}")
        print(f"[INFO] Labels: {labels_dict}")
        
        # Update instance
        self.model = model
        self.labels_dict = labels_dict

    
    def recognize_from_landmarks(self, hand_landmarks):
        """
        Recognize letter from MediaPipe hand landmarks object
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            tuple: (letter, confidence) or (None, 0)
        """
        if self.model is None:
            return None, 0.0
        
        if hand_landmarks is None:
            return None, 0.0
        
        try:
            # Extract features
            features = self.extract_hand_features(hand_landmarks)
            
            # Predict
            prediction = self.model.predict([features])
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([features])[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0  # Default confidence
            
            # Get predicted label
            predicted_class = int(prediction[0]) if isinstance(prediction[0], (int, np.integer)) else prediction[0]
            
            # Map to letter
            if predicted_class in self.labels_dict:
                letter = self.labels_dict[predicted_class]
            elif str(predicted_class) in self.labels_dict:
                letter = self.labels_dict[str(predicted_class)]
            else:
                letter = str(predicted_class).upper()
            
            return letter, confidence
            
        except Exception as e:
            print(f"[ERROR] Recognition error: {e}")
            return None, 0.0
    
    def recognize_with_stability(self, hand_landmarks):
        """
        Recognize letter with stability check (must be consistent across frames)
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            tuple: (letter, confidence, is_stable)
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_recognition_time < self.cooldown_period:
            return self.last_recognized, 0.0, False
        
        letter, confidence = self.recognize_from_landmarks(hand_landmarks)
        
        if letter:
            self.gesture_history.append(letter)
            
            # Keep only recent history
            if len(self.gesture_history) > self.stability_frames:
                self.gesture_history.pop(0)
            
            # Check if gesture is stable - STRICTER requirements
            if len(self.gesture_history) >= self.stability_frames:
                from collections import Counter
                most_common = Counter(self.gesture_history).most_common(1)[0]
                
                # Require 90% consistency (was allowing 1 outlier in 5, now 2 outliers in 15)
                required_consistency = int(self.stability_frames * 0.87)  # 13 out of 15
                if most_common[1] >= required_consistency:
                    stable_letter = most_common[0]
                    
                    # Only return if different from last recognition
                    if stable_letter != self.last_recognized:
                        self.last_recognized = stable_letter
                        self.last_recognition_time = current_time
                        self.gesture_history.clear()
                        return stable_letter, confidence, True
        else:
            # No recognition, clear history
            if len(self.gesture_history) > 0:
                self.gesture_history.clear()
        
        return letter, confidence, False

    
    def recognize_dynamic_gesture(self, hand_landmarks):
        """
        Recognize dynamic gestures (words/phrases) by accumulating frames
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            tuple: (word, confidence, is_complete) where is_complete=True when buffer is full
        """
        if self.model is None or not self.dynamic_classes:
            return None, 0.0, False
        
        # Extract features from current frame
        features = self.extract_hand_features(hand_landmarks)
        if features is None:
            return None, 0.0, False
        
        # Add to buffer
        self.dynamic_gesture_buffer.append(features)
        
        # Limit buffer size
        if len(self.dynamic_gesture_buffer) > self.dynamic_buffer_max_frames:
            self.dynamic_gesture_buffer.pop(0)
        
        # Need minimum frames for word recognition
        min_frames_for_word = 15  # At least 0.5 seconds of gesture
        if len(self.dynamic_gesture_buffer) < min_frames_for_word:
            return None, 0.0, False
        
        # Average features across buffer (temporal aggregation)
        avg_features = np.mean(self.dynamic_gesture_buffer, axis=0)
        
        # Predict
        try:
            prediction_idx = self.model.predict([avg_features])[0]
            prediction = self.labels_dict.get(prediction_idx, str(prediction_idx))
            
            # Get confidence
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([avg_features])[0]
                confidence = float(np.max(probabilities)) * 100
            else:
                confidence = 50.0
            
            # Only return dynamic classes (words/phrases)
            if prediction in self.dynamic_classes and confidence >= self.dynamic_recognition_threshold * 100:
                # Check if buffer is "stable" (enough frames accumulated)
                is_complete = len(self.dynamic_gesture_buffer) >= 30  # ~1 second at 30fps
                return prediction, confidence, is_complete
            
        except Exception as e:
            print(f"[ERROR] Dynamic recognition failed: {e}")
        
        return None, 0.0, False
    
    def reset_dynamic_buffer(self):
        """Clear dynamic gesture buffer - call when gesture sequence ends"""
        self.dynamic_gesture_buffer = []

    
    def build_feature_database(self):
        """
        Legacy method for compatibility - trains model from scratch
        """
        print("[INFO] Building feature database (training new model)...")
        self.create_dataset_from_images()
        self.train_model()
    
    def close(self):
        """Release resources"""
        if hasattr(self, 'hands'):
            self.hands.close()
        print("[INFO] AlphabetClassifier closed")


if __name__ == "__main__":
    # Demo script
    print("\n" + "="*60)
    print("Sign Language Alphabet Classifier - Setup Script")
    print("="*60)
    
    classifier = AlphabetClassifier()
    
    print("\nAvailable commands:")
    print("1. Collect training images (run sign-language-detector-python/collect_imgs.py)")
    print("2. Create dataset from images: classifier.create_dataset_from_images()")
    print("3. Train model: classifier.train_model()")
    
    if classifier.model:
        print(f"\n✓ Model loaded with {len(classifier.labels_dict)} classes:")
        print(f"  Labels: {classifier.labels_dict}")
    else:
        print("\n⚠ No model found. To train a new model:")
        print("  1. First, collect images using collect_imgs.py")
        print("  2. Then run this script with training flag")

