"""
Dataset Creation Script - Professional Multi-Type Support
Creates training datasets from collected data with intelligent classification:
- Processes STATIC images (alphabet, numbers) 
- Processes DYNAMIC videos (words, phrases, sentences)
- Enhanced 3D feature extraction matching alphabet_classifier.py
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

import pickle
import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Professional data structure paths
BASE_DIR = './sign-language-detector-python'
STATIC_PATHS = {
    'alphabet': os.path.join(BASE_DIR, 'data', 'static', 'alphabet'),
    'numbers': os.path.join(BASE_DIR, 'data', 'static', 'numbers')
}

DYNAMIC_PATHS = {
    'words': os.path.join(BASE_DIR, 'data', 'dynamic', 'words'),
    'phrases': os.path.join(BASE_DIR, 'data', 'dynamic', 'phrases'),
    'sentences': os.path.join(BASE_DIR, 'data', 'dynamic', 'sentences')
}

# Legacy path for backward compatibility
LEGACY_DATA_DIR = os.path.join(BASE_DIR, 'data')

print("="*70)
print("PROFESSIONAL DATASET CREATION SYSTEM")
print("="*70)
print("\n📂 Supported data types:")
print("   STATIC: alphabet (A-Z), numbers (0-9)")
print("   DYNAMIC: words, phrases, sentences")
print()

# Check which data exists
available_data = []
data_sources = []

# Check new structure - STATIC
for data_type, path in STATIC_PATHS.items():
    if os.path.exists(path):
        # Check if there are actual class folders (A, B, C, etc.)
        class_folders = [d for d in os.listdir(path) 
                        if os.path.isdir(os.path.join(path, d)) 
                        and d not in ['.git', '__pycache__']]
        if class_folders:
            available_data.append(f"static/{data_type}")
            data_sources.append((path, data_type, 'static'))
            print(f"✓ Found: {data_type} with {len(class_folders)} classes in {path}")

# Check new structure - DYNAMIC
for data_type, path in DYNAMIC_PATHS.items():
    if os.path.exists(path):
        class_folders = [d for d in os.listdir(path) 
                        if os.path.isdir(os.path.join(path, d)) 
                        and d not in ['.git', '__pycache__']]
        if class_folders:
            available_data.append(f"dynamic/{data_type}")
            data_sources.append((path, data_type, 'dynamic'))
            print(f"✓ Found: {data_type} with {len(class_folders)} classes in {path}")

# Check legacy structure (old flat folder) - but skip if new structure exists
if not data_sources and os.path.exists(LEGACY_DATA_DIR):
    legacy_items = [d for d in os.listdir(LEGACY_DATA_DIR) 
                    if os.path.isdir(os.path.join(LEGACY_DATA_DIR, d)) 
                    and d not in ['static', 'dynamic', '.git', '__pycache__']
                    and len(os.listdir(os.path.join(LEGACY_DATA_DIR, d))) > 0]
    if legacy_items:
        available_data.append("legacy/data")
        data_sources.append((LEGACY_DATA_DIR, 'legacy', 'static'))
        print(f"✓ Found: legacy data with {len(legacy_items)} classes")

if not data_sources:
    print("\n[ERROR] No training data found!")
    print("[INFO] Run collect_data.py first to collect training images")
    print(f"[INFO] Expected locations:")
    for dtype, path in STATIC_PATHS.items():
        print(f"  - {path}")
    for dtype, path in DYNAMIC_PATHS.items():
        print(f"  - {path}")
    exit(1)

print(f"\n[INFO] Processing {len(data_sources)} data source(s)...")
print("[INFO] Using enhanced 3D feature extraction (X, Y, Z with scaling)")

data = []
labels = []
skipped = 0
total_processed = 0

# Process each data source
for data_path, data_type, mode in data_sources:
    print(f"\n{'='*70}")
    print(f"Processing: {data_type.upper()} ({mode.upper()}) from {data_path}")
    print(f"{'='*70}")
    
    for dir_ in os.listdir(data_path):
        dir_full_path = os.path.join(data_path, dir_)
        if not os.path.isdir(dir_full_path):
            continue
        
        # Skip system folders
        if dir_ in ['static', 'dynamic', '.git', '__pycache__']:
            continue
        
        print(f"[INFO] Processing class: {dir_}")
        count = 0
        
        if mode == 'dynamic':
            # Process VIDEO files
            for video_name in os.listdir(dir_full_path):
                if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue
                    
                video_path = os.path.join(dir_full_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    skipped += 1
                    continue
                
                frame_features = []
                frame_count = 0
                
                # Extract features from each frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            data_aux = []
                            x_ = []
                            y_ = []
                            z_ = []
                            
                            for landmark in hand_landmarks.landmark:
                                x_.append(landmark.x)
                                y_.append(landmark.y)
                                z_.append(landmark.z)
                            
                            min_x, max_x = min(x_), max(x_)
                            min_y, max_y = min(y_), max(y_)
                            min_z, max_z = min(z_), max(z_)
                            
                            range_x = max_x - min_x if max_x - min_x > 0.001 else 0.001
                            range_y = max_y - min_y if max_y - min_y > 0.001 else 0.001
                            range_z = max_z - min_z if max_z - min_z > 0.001 else 0.001
                            
                            for i in range(len(hand_landmarks.landmark)):
                                data_aux.append((x_[i] - min_x) / range_x)
                                data_aux.append((y_[i] - min_y) / range_y)
                                data_aux.append((z_[i] - min_z) / range_z)
                            
                            frame_features.append(data_aux)
                            frame_count += 1
                            break  # Only process first hand per frame
                
                cap.release()
                
                # Use average features across all frames for now
                if frame_features:
                    avg_features = np.mean(frame_features, axis=0).tolist()
                    data.append(avg_features)
                    labels.append(dir_)
                    count += 1
                else:
                    skipped += 1
        
        else:
            # Process IMAGE files (existing code)
            for img_name in os.listdir(dir_full_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(dir_full_path, img_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # ENHANCED 3D FEATURE EXTRACTION
                        data_aux = []
                        x_ = []
                        y_ = []
                        z_ = []
                        
                        # Collect all coordinates
                        for landmark in hand_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)
                            z_.append(landmark.z)
                        
                        # Calculate ranges for scaling
                        min_x, max_x = min(x_), max(x_)
                        min_y, max_y = min(y_), max(y_)
                        min_z, max_z = min(z_), max(z_)
                    
                    range_x = max_x - min_x if max_x - min_x > 0.001 else 0.001
                    range_y = max_y - min_y if max_y - min_y > 0.001 else 0.001
                    range_z = max_z - min_z if max_z - min_z > 0.001 else 0.001
                    
                    # Normalize AND scale to [0, 1]
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append((x_[i] - min_x) / range_x)
                        data_aux.append((y_[i] - min_y) / range_y)
                        data_aux.append((z_[i] - min_z) / range_z)
                    
                    data.append(data_aux)
                    labels.append(dir_)
                    count += 1
                else:
                    skipped += 1
        
        if count > 0:
            print(f"  ✓ Processed {count} images for '{dir_}'")
            total_processed += count

# Save enhanced dataset
output_file = os.path.join('built-in', 'dataset', 'data.pickle')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if len(data) == 0:
    print("\n[ERROR] No valid training data found!")
    print("[INFO] Possible reasons:")
    print("  1. No images collected yet - run collect_data.py")
    print("  2. Images don't contain detectable hands")
    print("  3. Wrong folder structure - check data paths")
    exit(1)

with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n{'='*70}")
print("✓ DATASET CREATION COMPLETE")
print(f"{'='*70}")
print(f"✓ Total samples: {len(data)} (skipped {skipped} invalid)")
print(f"✓ Feature size: {len(data[0])} features per sample (21 landmarks × 3 coords)")
print(f"✓ Unique classes: {len(set(labels))} - {sorted(set(labels))}")
print(f"✓ Saved to: {output_file}")
print(f"\n📊 Class distribution:")
for label in sorted(set(labels)):
    count = labels.count(label)
    print(f"   {label}: {count} samples")
print(f"\n[INFO] Next step: Run train_model.py to train the classifier")
print("="*70)
