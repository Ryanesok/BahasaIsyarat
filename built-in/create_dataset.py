"""
Dataset Creation Script - Professional Multi-Type Support
Creates training datasets from collected data with intelligent classification:
- Processes STATIC images (alphabet, numbers) 
- Processes DYNAMIC videos (words, phrases, sentences)
- Enhanced 3D feature extraction matching alphabet_classifier.py
"""

import os
import sys

# Suppress only TensorFlow/MediaPipe verbose logs (keep actual errors visible)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Changed from 3 to 2 to show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'  # Changed from 3 to 2

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pickle
import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp

# Import path configuration
try:
    from path_config import PROJECT_ROOT, DATA_STRUCTURE
except ImportError:
    # Fallback if running standalone
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_STRUCTURE = os.path.join(PROJECT_ROOT, 'sign-language-detector-python')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, max_num_hands=1)

# Professional data structure paths
BASE_DIR = DATA_STRUCTURE

# Batch processing to avoid memory issues
BATCH_SIZE = 50  # Process in batches to prevent memory overflow

print("="*70)
print("PROFESSIONAL DATASET CREATION SYSTEM")
print("="*70)
print("\nSearching for training data recursively...")
print(f"Base directory: {BASE_DIR}")
print()

# Recursive search for data folders
def find_data_folders(root_path, max_depth=4):
    """Recursively search for folders containing training data"""
    data_folders = []
    
    def search_recursive(path, current_depth=0):
        if current_depth > max_depth:
            return
        
        if not os.path.exists(path):
            return
        
        try:
            items = os.listdir(path)
        except PermissionError:
            return
        
        # Check if this folder contains image/video files
        has_media = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')) 
                       for f in items if os.path.isfile(os.path.join(path, f)))
        
        if has_media:
            # This is a leaf folder with actual data
            parent_name = os.path.basename(path)
            grandparent_path = os.path.dirname(path)
            grandparent_name = os.path.basename(grandparent_path)
            
            # Determine type based on path
            if 'static' in grandparent_path.lower() or 'alphabet' in grandparent_path.lower() or 'numbers' in grandparent_path.lower():
                data_type = 'static'
            elif 'dynamic' in grandparent_path.lower() or 'word' in grandparent_path.lower():
                data_type = 'dynamic'
            else:
                # Assume static if only images
                is_video = any(f.lower().endswith(('.mp4', '.avi', '.mov')) 
                             for f in items if os.path.isfile(os.path.join(path, f)))
                data_type = 'dynamic' if is_video else 'static'
            
            data_folders.append({
                'path': os.path.dirname(path),  # Parent folder containing all classes
                'class': parent_name,
                'type': data_type,
                'full_path': path
            })
        else:
            # Continue searching in subdirectories
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and item not in ['.git', '__pycache__', 'logs']:
                    search_recursive(item_path, current_depth + 1)
    
    search_recursive(root_path)
    return data_folders

# Find all data folders
found_folders = find_data_folders(BASE_DIR)

# Group by parent path
data_sources = {}
for folder_info in found_folders:
    parent_path = folder_info['path']
    if parent_path not in data_sources:
        data_sources[parent_path] = {
            'type': folder_info['type'],
            'classes': []
        }
    data_sources[parent_path]['classes'].append(folder_info['class'])

available_data = []
organized_sources = []

for parent_path, info in data_sources.items():
    category = os.path.basename(parent_path)
    data_type = info['type']
    classes = info['classes']
    
    available_data.append(f"{data_type}/{category}")
    organized_sources.append((parent_path, category, data_type))
    print(f"[OK] Found: {category} ({data_type}) with {len(classes)} classes")
    print(f"  Path: {parent_path}")
    print(f"  Classes: {', '.join(sorted(classes)[:10])}" + ("..." if len(classes) > 10 else ""))
    print()

if not organized_sources:
    print("\n" + "="*70)
    print("[ERROR] NO TRAINING DATA FOUND!")
    print("="*70)
    print(f"\n[INFO] Searched in: {BASE_DIR}")
    print(f"[INFO] Base directory exists: {os.path.exists(BASE_DIR)}")
    print("\n[ACTION REQUIRED] Please collect training data first:")
    print("  1. Run collect_data.py to collect images")
    print("  2. Ensure data is saved to sign-language-detector-python/data/")
    print("\n[INFO] Searched recursively up to 4 levels deep")
    print("="*70)
    sys.stdout.flush()
    sys.exit(1)

print(f"\n[INFO] Processing {len(organized_sources)} data source(s)...")
print("[INFO] Using enhanced 3D feature extraction (X, Y, Z with scaling)")

data = []
labels = []
skipped = 0
total_processed = 0

# Process each data source
for data_path, data_type, mode in organized_sources:
    print(f"\n{'='*70}")
    print(f"Processing: {data_type.upper()} ({mode.upper()}) from {data_path}")
    print(f"{'='*70}")
    
    prev_skipped = skipped
    
    for dir_ in os.listdir(data_path):
        dir_full_path = os.path.join(data_path, dir_)
        if not os.path.isdir(dir_full_path):
            continue
        
        # Skip system folders
        if dir_ in ['static', 'dynamic', '.git', '__pycache__']:
            continue
        
        print(f"[INFO] Processing class: {dir_}")
        sys.stdout.flush()  # Force output to display immediately
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
            # Process IMAGE files with batch control
            img_files = [f for f in os.listdir(dir_full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_imgs = len(img_files)
            processed_imgs = 0
            
            for img_name in img_files:
                processed_imgs += 1
                
                # Progress indicator
                if processed_imgs % 20 == 0 or processed_imgs == total_imgs:
                    print(f"  ... {processed_imgs}/{total_imgs}", end='\r')
                    sys.stdout.flush()
                
                # Batch checkpoint - force garbage collection
                if processed_imgs % BATCH_SIZE == 0:
                    import gc
                    gc.collect()
                
                img_path = os.path.join(dir_full_path, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        skipped += 1
                        continue
                    
                    # Resize large images to prevent memory issues
                    h, w = img.shape[:2]
                    if h > 1280 or w > 1280:
                        scale = 1280 / max(h, w)
                        img = cv2.resize(img, None, fx=scale, fy=scale)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]  # Take first hand only
                        
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
                        
                except Exception as e:
                    # Skip problematic images
                    skipped += 1
                    continue
            
            # Clear progress line
            print(" " * 30, end='\r')
            sys.stdout.flush()
        
        if count > 0:
            print(f"  [OK] Processed {count} images for '{dir_}'")
            sys.stdout.flush()
            total_processed += count
        else:
            print(f"  ⚠ No valid data found for '{dir_}' (skipped {skipped - prev_skipped} files)")
            sys.stdout.flush()
        
        prev_skipped = skipped

# Save enhanced dataset using absolute path from path_config
output_file = os.path.join(PROJECT_ROOT, 'built-in', 'dataset', 'data.pickle')
os.makedirs(os.path.dirname(output_file), exist_ok=True)
print(f"\n[DEBUG] Saving dataset to: {output_file}")
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")
sys.stdout.flush()

if len(data) == 0:
    print("\n" + "="*70)
    print("[ERROR] NO VALID HAND LANDMARKS DETECTED!")
    print("="*70)
    print(f"\n[INFO] Processed {total_processed} files but found 0 valid hand landmarks")
    print(f"[INFO] Skipped {skipped} files")
    print("\n[POSSIBLE CAUSES]:")
    print("  1. Images don't contain visible hands")
    print("  2. Hand gestures are unclear or partially visible")
    print("  3. Poor lighting conditions in images")
    print("  4. Images are corrupted")
    print("\n[SOLUTION]:")
    print("  → Run collect_data.py again with better hand visibility")
    print("  → Ensure good lighting and clear hand gestures")
    print("  → Check that hands are fully in frame")
    print("="*70)
    hands.close()
    sys.stdout.flush()
    sys.exit(1)

with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n{'='*70}")
print("[SUCCESS] DATASET CREATION COMPLETE")
print("="*70)
print(f"Total samples: {len(data)} (skipped {skipped} invalid)")
print(f"Feature size: {len(data[0])} features per sample (21 landmarks x 3 coords)")
print(f"Unique classes: {len(set(labels))} - {sorted(set(labels))}")
print(f"Saved to: {output_file}")
print(f"\nClass distribution:")
for label in sorted(set(labels)):
    count = labels.count(label)
    print(f"   {label}: {count} samples")
print(f"\n[INFO] Next step: Run train_model.py to train the classifier")
print("="*70)

# Cleanup MediaPipe resources
hands.close()
sys.stdout.flush()
sys.exit(0)  # Explicit successful exit
