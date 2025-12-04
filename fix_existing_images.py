"""
Fix Existing Images - Remove Drawn Landmarks
This script processes already-collected images that have drawn landmarks
and recreates clean versions suitable for training.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Suppress warnings
if sys.platform == 'win32':
    stderr_fd = sys.stderr.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    sys.stderr = open(os.devnull, 'w')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*70)
print("IMAGE LANDMARK REMOVAL TOOL")
print("="*70)
print("\nThis will attempt to clean images with drawn landmarks.")
print("Original images will be backed up to _backup folder.\n")

BASE_DIR = './sign-language-detector-python'
STATIC_PATHS = {
    'alphabet': os.path.join(BASE_DIR, 'data', 'static', 'alphabet'),
    'numbers': os.path.join(BASE_DIR, 'data', 'static', 'numbers')
}

def remove_green_yellow_overlay(image):
    """
    Remove green and yellow drawn landmarks from image
    Uses color masking and inpainting
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Green color range (landmarks)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Yellow color range (connections)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_green, mask_yellow)
    
    # Dilate mask slightly to cover line thickness
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Inpaint to remove the markings
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return result

# Process each data source
total_processed = 0
total_cleaned = 0

for data_type, path in STATIC_PATHS.items():
    if not os.path.exists(path):
        continue
    
    print(f"\n{'='*70}")
    print(f"Processing: {data_type.upper()}")
    print(f"{'='*70}")
    
    for class_folder in os.listdir(path):
        class_path = os.path.join(path, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        backup_path = os.path.join(class_path, '_backup')
        
        # Check if already processed
        if os.path.exists(backup_path):
            response = input(f"\n'{class_folder}' already has backup. Skip? (y/n): ")
            if response.lower() == 'y':
                print(f"Skipping {class_folder}")
                continue
        else:
            os.makedirs(backup_path, exist_ok=True)
        
        print(f"\n[INFO] Processing class: {class_folder}")
        
        count = 0
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_name)
            backup_img_path = os.path.join(backup_path, img_name)
            
            # Skip if already backed up
            if os.path.exists(backup_img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Backup original
            cv2.imwrite(backup_img_path, img)
            
            # Clean the image
            cleaned = remove_green_yellow_overlay(img)
            
            # Save cleaned version
            cv2.imwrite(img_path, cleaned)
            
            count += 1
            total_processed += 1
            
            if count % 50 == 0:
                print(f"  Progress: {count} images...")
        
        if count > 0:
            print(f"  ✓ Cleaned {count} images for '{class_folder}'")
            total_cleaned += count

print(f"\n{'='*70}")
print("✓ CLEANING COMPLETE")
print(f"{'='*70}")
print(f"✓ Total images processed: {total_processed}")
print(f"✓ Originals backed up in each class/_backup folder")
print(f"\n[INFO] Now run: python train_model.py")
print("="*70)
