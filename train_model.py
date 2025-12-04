"""
Model Training Script
Trains a RandomForestClassifier on collected sign language data
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

# Add built-in folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'built-in'))

from alphabet_classifier import AlphabetClassifier

print("="*60)
print("Sign Language Model Training")
print("="*60)

print("\nStep 1: Creating dataset from collected images...")
print("[INFO] Running built-in/create_dataset.py...")
import subprocess
result = subprocess.run([sys.executable, 'built-in/create_dataset.py'], 
                       capture_output=False, text=True)

if result.returncode != 0:
    print("[ERROR] Dataset creation failed!")
    exit(1)

print("\nStep 2: Training RandomForestClassifier...")

# Check if data.pickle exists in built-in/dataset
dataset_path = os.path.join('built-in', 'dataset', 'data.pickle')
if not os.path.exists(dataset_path):
    print(f"[ERROR] {dataset_path} not found after dataset creation!")
    exit(1)

# Load and train directly instead of using classifier method
print(f"[INFO] Loading dataset from {dataset_path}...")
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open(dataset_path, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"[INFO] Dataset loaded: {len(data)} samples, {len(set(labels))} classes")
print(f"[INFO] Classes: {sorted(set(labels))}")

# Separate static (single letters) and dynamic (words) classes
static_classes = set()
dynamic_classes = set()
for label in set(labels):
    if len(label) == 1 or label.isdigit():
        static_classes.add(label)
    else:
        dynamic_classes.add(label)

print(f"\n[INFO] Class types:")
print(f"  - Static (letters/numbers): {len(static_classes)} classes - {sorted(static_classes)}")
print(f"  - Dynamic (words/phrases): {len(dynamic_classes)} classes - {sorted(dynamic_classes)}")

# Check for classes with insufficient samples
from collections import Counter
class_counts = Counter(labels)
min_samples_needed = 2  # Minimum for stratified split

print(f"\n[INFO] Class distribution:")
insufficient_classes = []
for cls, count in sorted(class_counts.items()):
    status = "⚠️ TOO FEW" if count < min_samples_needed else "✓"
    print(f"  {status} {cls}: {count} samples")
    if count < min_samples_needed:
        insufficient_classes.append((cls, count))

if insufficient_classes:
    print(f"\n⚠️  WARNING: {len(insufficient_classes)} class(es) have too few samples:")
    for cls, count in insufficient_classes:
        print(f"    - '{cls}': {count} sample(s) (need at least {min_samples_needed})")
    print(f"\n[INFO] Continuing with non-stratified split...")
    use_stratify = False
else:
    use_stratify = True

if len(data) == 0:
    print("[ERROR] No data in dataset!")
    exit(1)

# Split dataset (stratified only if all classes have enough samples)
if use_stratify:
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )
else:
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True
    )

# Train model with optimized parameters
print("[INFO] Training RandomForestClassifier (200 trees, depth 25)...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

# Test accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

# CRITICAL VALIDATION: Check model has all expected classes
trained_classes = set(model.classes_)
expected_classes = set(labels)
if trained_classes != expected_classes:
    print(f"\n[ERROR] Model training incomplete!")
    print(f"  Expected classes: {sorted(expected_classes)}")
    print(f"  Trained classes:  {sorted(trained_classes)}")
    print(f"  Missing classes:  {sorted(expected_classes - trained_classes)}")
    exit(1)

print(f"\n{'='*60}")
print(f"✓ MODEL TRAINING COMPLETE")
print(f"{'='*60}")
print(f"✓ Training accuracy: {score * 100:.2f}%")
print(f"✓ Feature dimensions: {len(data[0])} (3D enhanced features)")
print(f"✓ Classes trained: {sorted(model.classes_)}")

# Save model to built-in/dataset directory
model_path = os.path.join('built-in', 'dataset', 'model.p')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model, 
        'labels_dict': {i: label for i, label in enumerate(set(labels))},
        'static_classes': sorted(list(static_classes)),
        'dynamic_classes': sorted(list(dynamic_classes))
    }, f)

print(f"✓ Model saved to: {model_path}")
print(f"\n[INFO] Next step: Run desktop_app.py to test the model")
print("="*60)

print("\n" + "="*60)
print("Training Complete!")
print("\nYour model is ready to use.")
print("Run desktop_app.py to test it.")
print("="*60)
