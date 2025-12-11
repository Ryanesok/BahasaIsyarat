""" 
Path Configuration Helper
DETECTS if running as .exe or .py and uses correct path
"""

import os
import sys


def get_application_path():
    """
    Get the project root directory
    - If running as .exe (frozen): use sys.executable directory
    - If running as .py: use __file__ directory
    - If .exe is in 'built-in' subfolder, go up one level
    """
    if getattr(sys, 'frozen', False):
        # Running as .exe - use sys.executable
        exe_dir = os.path.dirname(sys.executable)
        
        # Check if we're in a 'built-in' subfolder
        if os.path.basename(exe_dir).lower() == 'built-in':
            # Go up one level to get the actual project root
            return os.path.dirname(exe_dir)
        
        return exe_dir
    else:
        # Running as .py - use __file__ directory (workspace folder)
        return os.path.dirname(os.path.abspath(__file__))


# ============================================================
# GLOBAL PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = get_application_path()
print(f"[PATH] Running as: {'EXE' if getattr(sys, 'frozen', False) else 'PYTHON SCRIPT'}")
print(f"[PATH] sys.executable: {sys.executable}")
print(f"[PATH] PROJECT_ROOT: {PROJECT_ROOT}")
BUILT_IN_DIR = os.path.join(PROJECT_ROOT, 'built-in')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data')
DATA_STRUCTURE = os.path.join(PROJECT_ROOT, 'sign-language-detector-python')  # For create_dataset.py

# Data directories
STATIC_ALPHABET_DIR = os.path.join(DATA_ROOT, 'static', 'alphabet')
STATIC_NUMBERS_DIR = os.path.join(DATA_ROOT, 'static', 'numbers')
DYNAMIC_WORDS_DIR = os.path.join(DATA_ROOT, 'dynamic', 'words')

# Dataset and model files
DATASET_DIR = os.path.join(BUILT_IN_DIR, 'dataset')
DATASET_FILE = os.path.join(DATASET_DIR, 'data.pickle')
MODEL_FILE = os.path.join(DATASET_DIR, 'model.p')

# Application files
DESKTOP_APP_FILE = os.path.join(PROJECT_ROOT, 'desktop_app.py')
COLLECT_DATA_FILE = os.path.join(PROJECT_ROOT, 'collect_data.py')
TRAIN_MODEL_FILE = os.path.join(PROJECT_ROOT, 'train_model.py')
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config.json')


def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        BUILT_IN_DIR,
        DATASET_DIR,
        DATA_ROOT,
        STATIC_ALPHABET_DIR,
        STATIC_NUMBERS_DIR,
        DYNAMIC_WORDS_DIR,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return True


# Auto-create directories on import
ensure_directories()
