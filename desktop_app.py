import os
import sys

# CRITICAL: Redirect stderr at file descriptor level BEFORE any imports
if sys.platform == 'win32':
    # Redirect at OS level to catch C++ output
    stderr_fd = sys.stderr.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    # Also redirect Python-level stderr
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

# Suppress MediaPipe C++ warnings by redirecting stderr temporarily
import contextlib
import io

class SuppressStderr:
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stderr = self.old_stderr

# Add built-in folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'built-in'))

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import threading
import time
import json

# Import modules with suppressed warnings
from hand_tracker import HandTracker
from dev_console import DeveloperConsole
from alphabet_classifier import AlphabetClassifier
from session_logger import SessionLogger
from autocorrect import AutoCorrect
from visual_helpers import draw_roi_guide, draw_state_indicator
from sensor_fusion import SensorFusion, GestureDecisionEngine

# ---------------------------------
# LOAD CONFIGURATION
# ---------------------------------
def load_config(config_path="config.json"):
    """Load application configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return {
            "mqtt": {"broker": "broker.hivemq.com", "port": 1883, 
                     "topic_gesture": "pameran/gerakan", "client_id": "kelompok-4",
                     "reconnect_interval": 3},
            "camera": {"device_id": 0, "width": 1280, "height": 720, "fps_limit": 45},
            "detection": {"min_detection_confidence": 0.5, "cooldown_seconds": 1.5,
                         "min_stable_frames": 3},
            "logging": {"enabled": True, "log_dir": "logs"},
            "autocorrect": {"enabled": True},
            "ui": {"show_roi_guide": True}
        }

CONFIG = load_config()

# Extract config values
MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_TOPIC_GESTURE = CONFIG["mqtt"]["topic_gesture"]
MQTT_TOPIC_VISUAL = CONFIG["mqtt"].get("topic_visual", "pameran/visual")
CLIENT_ID = CONFIG["mqtt"]["client_id"]
RECONNECT_INTERVAL = CONFIG["mqtt"]["reconnect_interval"]
# ---------------------------------

class gestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SIGN LANGUAGE READER")
        
        #-------------------------------------
        self.conn_status = tk.StringVar(value="Connecting...")
        self.visual_gestur = tk.StringVar(value="NOT CALIBRATED")
        self.move_gesture = tk.StringVar(value="--")
        self.fps_display = tk.StringVar(value="FPS: 0")
        self.detection_confidence = tk.StringVar(value="Confidence: 0%")
        self.app_state = tk.StringVar(value="IDLE")  # State: IDLE, LISTENING, COOLDOWN
        
        # Translation variables
        self.detected_letter = tk.StringVar(value="--")
        self.current_word = tk.StringVar(value="")
        self.sentence_text = tk.StringVar(value="")
        self.correction_suggestion = tk.StringVar(value="")

        self.running = True
        self.camera_ready = False
        self.image_queue = None  # Will hold the latest frame for GUI update
        
        # FPS control from config
        self.fps_limit = CONFIG["camera"]["fps_limit"]
        self.frame_delay = 1.0 / self.fps_limit
        self.last_frame_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Detection state machine
        self.detection_state = "IDLE"  # IDLE, LISTENING, COOLDOWN
        self.last_detection_time = 0
        self.cooldown_duration = CONFIG["detection"]["cooldown_seconds"]
        self.stable_detection_frames = 0
        self.min_stable_frames = CONFIG["detection"]["min_stable_frames"]
        self.last_detected_letter = None
        
        # MQTT reconnection
        self.mqtt_connected = False
        self.mqtt_reconnect_timer = None
        #-------------------------------------
        
        #-------------------------------------
        self.tracker = HandTracker()
        self.alphabet_classifier = AlphabetClassifier()
        self.dev_console = None  # Will be created after GUI
        
        # Initialize sensor fusion and decision engine
        self.sensor_fusion = SensorFusion(camera_weight=0.5, mpu_weight=0.5)
        self.gesture_decision = GestureDecisionEngine(min_confidence=60.0, stability_frames=3)
        
        # Initialize autocorrect and logging
        self.autocorrect = AutoCorrect(
            enabled=CONFIG["autocorrect"].get("enabled", True)
        )
        self.session_logger = SessionLogger(
            log_dir=CONFIG["logging"].get("log_dir", "logs"),
            enabled=CONFIG["logging"].get("enabled", True)
        )
        
        self.frame_count = 0
        self.detection_count = 0
        self.show_roi_guide = CONFIG["ui"].get("show_roi_guide", True)
        
        self.create_widgets()
        
        # Defer camera and MQTT initialization to after GUI is shown
        self.root.after(100, self.initialize_camera)
        self.root.after(200, self.setup_mqtt)
        self.root.after(300, self.create_dev_console)  # Create dev console
        self.root.after(400, self.load_alphabet_database)  # Load alphabet features
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        #-------------------------------------
        
    def create_widgets(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # ========== LEFT SIDE: VIDEO FEED (70% width) ==========
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        video_container = ttk.Frame(left_frame, relief=tk.RIDGE, borderwidth=3)
        video_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_container, text="[CAMERA] Initializing Camera...\n\nPlease wait...", 
                                     font=("Helvetica", 12), anchor=tk.CENTER)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # ========== RIGHT SIDE: USER INFORMATION (30% width) ==========
        right_frame = tk.Frame(main_frame, width=380, bg="#1a1a1a")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Scrollable container for right panel
        canvas = tk.Canvas(right_frame, bg="#1a1a1a", highlightthickness=0, width=360)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a1a", width=360)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=360)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Inner padding frame with reduced padding
        inner_frame = tk.Frame(scrollable_frame, bg="#1a1a1a", padx=10, pady=10)
        inner_frame.pack(fill=tk.BOTH, expand=True)
        
        # ========== SECTION 1: TRANSLATION OUTPUT ==========
        trans_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        trans_section.pack(fill=tk.X, pady=(0, 10))
        
        trans_inner = tk.Frame(trans_section, bg="#2b2b2b", padx=8, pady=8)
        trans_inner.pack(fill=tk.BOTH, expand=True)
        
        trans_header = tk.Label(trans_inner, text="[TRANSLATION]", 
                               font=("Helvetica", 10, "bold"), fg="#00ff88", bg="#2b2b2b")
        trans_header.pack(anchor=tk.W)
        
        tk.Frame(trans_inner, height=1, bg="#00ff88").pack(fill=tk.X, pady=(2, 6))
        
        # Detected Letter Display
        letter_label = tk.Label(trans_inner, text="Letter:", 
                               font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b")
        letter_label.pack(anchor=tk.W)
        
        self.letter_display = tk.Label(trans_inner, textvariable=self.detected_letter, 
                                      font=("Helvetica", 36, "bold"), fg="#00ffff", bg="#2b2b2b")
        self.letter_display.pack(anchor=tk.W, pady=(0, 6))
        
        # Word Buffer
        word_label = tk.Label(trans_inner, text="Word:", 
                             font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b")
        word_label.pack(anchor=tk.W)
        
        self.word_display = tk.Label(trans_inner, textvariable=self.current_word, 
                                    font=("Courier", 14, "bold"), fg="#ffff00", bg="#1a1a1a", 
                                    anchor=tk.W, padx=8, pady=6, relief=tk.SUNKEN)
        self.word_display.pack(fill=tk.X, pady=(0, 6))
        
        # Control Buttons
        btn_frame = tk.Frame(trans_inner, bg="#2b2b2b")
        btn_frame.pack(fill=tk.X, pady=(0, 6))
        
        self.space_btn = tk.Button(btn_frame, text="SPC", font=("Helvetica", 8, "bold"),
                                   bg="#444444", fg="#ffffff", activebackground="#666666",
                                   command=self.add_space, padx=8, pady=3)
        self.space_btn.pack(side=tk.LEFT, padx=(0, 3))
        
        self.backspace_btn = tk.Button(btn_frame, text="<-", font=("Helvetica", 8, "bold"),
                                       bg="#664444", fg="#ffffff", activebackground="#886666",
                                       command=self.backspace, padx=8, pady=3)
        self.backspace_btn.pack(side=tk.LEFT, padx=(0, 3))
        
        self.clear_btn = tk.Button(btn_frame, text="CLR", font=("Helvetica", 8, "bold"),
                                   bg="#664444", fg="#ffffff", activebackground="#886666",
                                   command=self.clear_all, padx=8, pady=3)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Sentence Output
        sentence_label = tk.Label(trans_inner, text="Sentence:", 
                                 font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b")
        sentence_label.pack(anchor=tk.W)
        
        self.sentence_display = tk.Label(trans_inner, textvariable=self.sentence_text, 
                                        font=("Arial", 10), fg="#ffffff", bg="#1a1a1a", 
                                        anchor=tk.W, justify=tk.LEFT, padx=8, pady=6, 
                                        relief=tk.SUNKEN, wraplength=320, height=3)
        self.sentence_display.pack(fill=tk.BOTH)
        
        # ========== SECTION 2: HAND DETECTION STATUS ==========
        detect_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        detect_section.pack(fill=tk.X, pady=(0, 10))
        
        detect_inner = tk.Frame(detect_section, bg="#2b2b2b", padx=8, pady=8)
        detect_inner.pack(fill=tk.BOTH)
        
        detect_header = tk.Label(detect_inner, text="[HAND DETECTION]", 
                                font=("Helvetica", 10, "bold"), fg="#ff9500", bg="#2b2b2b")
        detect_header.pack(anchor=tk.W)
        
        tk.Frame(detect_inner, height=1, bg="#ff9500").pack(fill=tk.X, pady=(2, 6))
        
        # Position display
        pos_label = tk.Label(detect_inner, text="Position:", 
                            font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b")
        pos_label.pack(anchor=tk.W)
        
        self.visual_label = tk.Label(detect_inner, textvariable=self.visual_gestur, 
                                    font=("Helvetica", 18, "bold"), fg="#00ffff", bg="#2b2b2b")
        self.visual_label.pack(anchor=tk.W, pady=(0, 3))
        
        self.confidence_label = tk.Label(detect_inner, textvariable=self.detection_confidence, 
                                        font=("Helvetica", 8), fg="#888888", bg="#2b2b2b")
        self.confidence_label.pack(anchor=tk.W, pady=(0, 6))
        
        # Tracking status
        track_status = tk.Label(detect_inner, text="MediaPipe Active", 
                               font=("Helvetica", 7), fg="#00ff00", bg="#2b2b2b")
        track_status.pack(anchor=tk.W)
        
        # ========== SECTION 3: MQTT CONNECTION ==========
        mqtt_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        mqtt_section.pack(fill=tk.X, pady=(0, 10))
        
        mqtt_inner = tk.Frame(mqtt_section, bg="#2b2b2b", padx=8, pady=8)
        mqtt_inner.pack(fill=tk.BOTH)
        
        mqtt_header = tk.Label(mqtt_inner, text="[MQTT]", 
                              font=("Helvetica", 10, "bold"), fg="#00d4ff", bg="#2b2b2b")
        mqtt_header.pack(anchor=tk.W)
        
        tk.Frame(mqtt_inner, height=1, bg="#00d4ff").pack(fill=tk.X, pady=(2, 6))
        
        self.conn_label = tk.Label(mqtt_inner, textvariable=self.conn_status, 
                                  font=("Helvetica", 8), fg="#ffaa00", bg="#2b2b2b", 
                                  wraplength=320, justify=tk.LEFT)
        self.conn_label.pack(anchor=tk.W, pady=(0, 6))
        
        move_label = tk.Label(mqtt_inner, text="Data:", 
                             font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b")
        move_label.pack(anchor=tk.W)
        
        self.movement_label = tk.Label(mqtt_inner, textvariable=self.move_gesture, 
                                      font=("Helvetica", 14, "bold"), fg="#ff9500", bg="#2b2b2b")
        self.movement_label.pack(anchor=tk.W)
        
        # ========== SECTION 4: PERFORMANCE ==========
        perf_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        perf_section.pack(fill=tk.X, pady=(0, 10))
        
        perf_inner = tk.Frame(perf_section, bg="#2b2b2b", padx=8, pady=8)
        perf_inner.pack(fill=tk.BOTH)
        
        perf_header = tk.Label(perf_inner, text="[PERFORMANCE]", 
                              font=("Helvetica", 10, "bold"), fg="#ffff00", bg="#2b2b2b")
        perf_header.pack(anchor=tk.W)
        
        tk.Frame(perf_inner, height=1, bg="#ffff00").pack(fill=tk.X, pady=(2, 6))
        
        self.fps_label = tk.Label(perf_inner, textvariable=self.fps_display, 
                                 font=("Courier", 9, "bold"), fg="#ffffff", bg="#2b2b2b")
        self.fps_label.pack(anchor=tk.W)
        
    def create_dev_console(self):
        """Create developer console window"""
        self.dev_console = DeveloperConsole(self)
        self.dev_console.add_log("Application started successfully")
        self.dev_console.add_log(f"MQTT Broker: {MQTT_BROKER}")
        self.dev_console.add_log("HandTracker: MediaPipe 21-point detection active")
    
    def load_alphabet_database(self):
        """Load alphabet feature database in background"""
        def load_db():
            try:
                # Check if model exists, if not provide instructions
                if self.alphabet_classifier.model is None:
                    if self.dev_console:
                        self.dev_console.add_log("No trained model found!", "WARN")
                        self.dev_console.add_log("To train a model:", "INFO")
                        self.dev_console.add_log("1. Run: python sign-language-detector-python/collect_imgs.py", "INFO")
                        self.dev_console.add_log("2. Then train using alphabet_classifier.py", "INFO")
                    print("[WARN] No trained model found. See console for training instructions.")
                else:
                    if self.dev_console:
                        labels_count = len(self.alphabet_classifier.labels_dict)
                        labels = ', '.join(self.alphabet_classifier.labels_dict.values())
                        self.dev_console.add_log(f"Alphabet model loaded: {labels_count} classes ({labels})")
                    print(f"[INFO] Alphabet classifier ready with {labels_count} classes")
            except Exception as e:
                print(f"[ERROR] Failed to initialize alphabet classifier: {e}")
                if self.dev_console:
                    self.dev_console.add_log(f"Alphabet classifier initialization failed: {e}", "ERROR")
        
        # Load in background thread
        threading.Thread(target=load_db, daemon=True).start()
    
    def add_space(self):
        """Add current word to sentence with autocorrect"""
        word = self.current_word.get().strip()
        if word:
            # Apply autocorrect if enabled
            if self.autocorrect.enabled:
                corrected = self.autocorrect.auto_correct(word)
                if corrected != word:
                    if self.dev_console:
                        self.dev_console.add_log(f"Autocorrect: '{word}' -> '{corrected}'")
                    word = corrected
            
            current_sentence = self.sentence_text.get()
            if current_sentence:
                self.sentence_text.set(current_sentence + " " + word)
            else:
                self.sentence_text.set(word)
            
            # Log word formation
            self.session_logger.log_word_formed(word)
            
            self.current_word.set("")
            self.correction_suggestion.set("")  # Clear suggestions
            if self.dev_console:
                self.dev_console.add_log(f"Word added to sentence: '{word}'")
    
    def backspace(self):
        """Remove last character from current word"""
        word = self.current_word.get()
        if word:
            self.current_word.set(word[:-1])
    
    def clear_all(self):
        """Clear all translation buffers"""
        self.current_word.set("")
        self.sentence_text.set("")
        self.detected_letter.set("--")
        if self.dev_console:
            self.dev_console.add_log("Translation cleared")
        
    def on_calibrate_press(self):
        # No longer needed - MediaPipe doesn't require calibration
        pass
    
    def _calculate_hand_movement(self, prev_landmarks, curr_landmarks):
        """
        Calculate hand movement between frames to detect dynamic gestures
        Returns normalized movement magnitude (0.0 to 1.0+)
        """
        if prev_landmarks is None or curr_landmarks is None:
            return 0.0
        
        try:
            # Calculate average displacement of key landmarks (wrist + fingertips)
            key_indices = [0, 4, 8, 12, 16, 20]  # Wrist + all fingertips
            total_movement = 0.0
            
            for idx in key_indices:
                prev_lm = prev_landmarks.landmark[idx]
                curr_lm = curr_landmarks.landmark[idx]
                
                dx = curr_lm.x - prev_lm.x
                dy = curr_lm.y - prev_lm.y
                dz = curr_lm.z - prev_lm.z
                
                # Euclidean distance
                movement = (dx**2 + dy**2 + dz**2) ** 0.5
                total_movement += movement
            
            # Average movement across key points
            avg_movement = total_movement / len(key_indices)
            return avg_movement
            
        except Exception as e:
            print(f"[ERROR] Movement calculation failed: {e}")
            return 0.0
    
    def initialize_camera(self):
        """Initialize camera in background after GUI is ready"""
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        print("[INFO] Camera initialization started")
        
    def setup_mqtt(self):
        """Initialize MQTT connection asynchronously with auto-reconnect"""
        def connect_mqtt():
            try:
                self.client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
                self.client.on_connect = self.on_connect
                self.client.on_message = self.on_message
                self.client.on_disconnect = self.on_disconnect
                self.root.after(0, lambda: self.conn_status.set("[...] Connecting to broker..."))
                self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
                self.client.loop_start()
                print("[INFO] MQTT connection initiated")
            except Exception as e:
                self.root.after(0, lambda: self.conn_status.set(f"Error: {e}"))
                self.root.after(0, lambda: self.conn_label.config(fg="#ff0000"))
                print(f"[ERROR] MQTT connection failed: {e}")
                # Schedule reconnect
                self.schedule_mqtt_reconnect()
        
        # Run MQTT connection in separate thread
        mqtt_thread = threading.Thread(target=connect_mqtt, daemon=True)
        mqtt_thread.start()
    
    def schedule_mqtt_reconnect(self):
        """Schedule automatic MQTT reconnection"""
        if self.mqtt_reconnect_timer:
            return  # Already scheduled
        
        def try_reconnect():
            if not self.mqtt_connected:
                print(f"[INFO] Attempting MQTT reconnection...")
                self.root.after(0, lambda: self.conn_status.set(f"[>>] Reconnecting..."))
                self.setup_mqtt()
            self.mqtt_reconnect_timer = None
        
        self.mqtt_reconnect_timer = self.root.after(RECONNECT_INTERVAL * 1000, try_reconnect)
    
    def on_disconnect(self, client, userdata, reason_code, properties):
        """Handle MQTT disconnection"""
        self.mqtt_connected = False
        self.conn_status.set(f"[!] Disconnected (code {reason_code})")
        self.conn_label.config(fg="#ffaa00")
        if self.dev_console:
            self.dev_console.add_log(f"MQTT disconnected: {reason_code}", "WARN")
        self.schedule_mqtt_reconnect()
        
    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.mqtt_connected = True
            self.conn_status.set(f"[OK] Connected to {MQTT_BROKER}")
            self.conn_label.config(fg="#00ff00")
            client.subscribe(MQTT_TOPIC_GESTURE)
            if self.dev_console:
                self.dev_console.add_log(f"MQTT connected to {MQTT_BROKER}")
                self.dev_console.update_stats(mqtt_status="Connected")
        else:
            self.mqtt_connected = False
            self.conn_status.set(f"[X] Connection Failed (code {reason_code})")
            self.conn_label.config(fg="#ff0000")
            if self.dev_console:
                self.dev_console.add_log(f"MQTT connection failed: {reason_code}", "ERROR")
                self.dev_console.update_stats(mqtt_status="Failed")
            self.schedule_mqtt_reconnect()
    
    def on_message(self, client, userdata, msg):
        if msg.topic == MQTT_TOPIC_GESTURE:
            try:
                payload = msg.payload.decode("utf-8").strip()
                
                # Try to parse as JSON (MPU sensor data)
                try:
                    import json
                    mpu_data = json.loads(payload)
                    
                    # Validate MPU data structure
                    if isinstance(mpu_data, dict) and 'AcX' in mpu_data:
                        # Update sensor fusion with MPU data
                        self.sensor_fusion.update_mpu_data(mpu_data)
                        
                        # Get MPU gesture
                        mpu_gesture, mpu_confidence = self.sensor_fusion.get_mpu_gesture()
                        
                        if mpu_gesture and mpu_confidence > 50:
                            self.move_gesture.set(f"{mpu_gesture} ({mpu_confidence:.0f}%)")
                            
                            if self.dev_console:
                                self.dev_console.add_log(
                                    f"MPU: {mpu_gesture} @ {mpu_confidence:.0f}% confidence"
                                )
                        
                        return  # MPU data processed
                        
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, treat as simple command
                    pass
                
                # Handle simple command messages (from ESP32 gestures)
                message = payload.upper()
                self.move_gesture.set(message)
                
                if self.dev_console:
                    self.dev_console.add_log(f"Received MQTT: {message}")
            
                if message in ["SPACE", "RIGHT", "TILT_RIGHT"]:
                    print(f"[MQTT] >> Action: SPACE TRIGGERED")
                    self.root.after(0, self.add_space)
                
                elif message in ["BACKSPACE", "LEFT", "TILT_LEFT"]:
                    print(f"[MQTT] >> Action: BACKSPACE TRIGGERED")
                    self.root.after(0, self.backspace)
                
                elif message in ["CLEAR", "SHAKE", "TILT_FORWARD"]:
                    print(f"[MQTT] >> Action: CLEAR TRIGGERED")
                    self.root.after(0, self.clear_all)
                    
            except Exception as e:
                print(f"[ERROR] MQTT message handling failed: {e}")
                if self.dev_console:
                    self.dev_console.add_log(f"MQTT error: {e}", "ERROR")
    def video_loop(self):
        # Lazy camera initialization
        try:
            # Get camera device from config
            camera_device = CONFIG["camera"]["device_id"]
            
            # Support for DirectShow device path (Windows)
            if isinstance(camera_device, str) and len(camera_device) > 10:
                # DirectShow path format
                self.cap = cv2.VideoCapture(camera_device, cv2.CAP_DSHOW)
                print(f"[INFO] Opening camera with DirectShow path: {camera_device}")
            else:
                # Standard device ID (integer)
                self.cap = cv2.VideoCapture(camera_device)
                print(f"[INFO] Opening camera with device ID: {camera_device}")
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
            if not self.cap.isOpened():
                self.root.after(0, lambda: self.video_label.config(text="[ERROR] Cannot open camera"))
                return
            self.camera_ready = True
        except Exception as e:
            print(f"Error opening camera: {e}")
            self.root.after(0, lambda: self.video_label.config(text=f"[ERROR] Camera Error:\n{e}"))
            return
            
        while self.running:
            # FPS control to prevent flickering
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_delay:
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate and update FPS
            self.fps_counter += 1
            if current_time - self.fps_start_time >= 1.0:
                fps = self.fps_counter / (current_time - self.fps_start_time)
                self.root.after(0, lambda f=fps: self.fps_display.set(f"FPS: {f:.1f}"))
                self.fps_counter = 0
                self.fps_start_time = current_time
            
            self.last_frame_time = current_time
                
            frame = cv2.flip(frame, 1)
            
            #---------------------------------------
            # Track hand using MediaPipe
            processing_start = time.time()
            frame_processed, gesture, confidence, hand_info = self.tracker.track(frame)
            processing_time = (time.time() - processing_start) * 1000  # ms
            
            # Add visual guides
            if self.show_roi_guide and gesture == "NOT DETECTED":
                frame_processed = draw_roi_guide(frame_processed)
            
            # Draw state indicator
            frame_processed = draw_state_indicator(frame_processed, self.detection_state)
            
            # Update frame counter
            self.frame_count += 1
            if gesture != "NOT DETECTED":
                self.detection_count += 1
            
            # ========== ALPHABET RECOGNITION WITH STATE MACHINE ==========
            landmarks = hand_info.get('landmarks', None)
            recognized_letter = None
            letter_confidence = 0.0
            
            # State machine logic
            current_time = time.time()
            
            if landmarks is not None and hasattr(self, 'alphabet_classifier'):
                try:
                    # Detect gesture type: static (letter) or dynamic (word)
                    # Dynamic gestures have more movement across frames
                    if hasattr(self, 'prev_hand_landmarks'):
                        # Calculate hand movement between frames
                        movement = self._calculate_hand_movement(self.prev_hand_landmarks, landmarks)
                        is_dynamic_gesture = movement > 0.05  # Threshold for dynamic gesture
                    else:
                        is_dynamic_gesture = False
                    
                    # Store current landmarks for next frame
                    self.prev_hand_landmarks = landmarks
                    
                    # === DYNAMIC GESTURE RECOGNITION (WORDS) ===
                    if is_dynamic_gesture and hasattr(self.alphabet_classifier, 'dynamic_classes') and self.alphabet_classifier.dynamic_classes:
                        word, word_conf, is_complete = self.alphabet_classifier.recognize_dynamic_gesture(landmarks)
                        
                        if word and is_complete:
                            # Word gesture recognized!
                            current_sentence = self.current_word.get()
                            self.current_word.set(current_sentence + " " + word + " ")
                            
                            # Log word detection
                            self.session_logger.log_detection(
                                self.frame_count, word, word_conf,
                                current_sentence + " " + word, self.sentence_text.get(),
                                hand_info.get('position', (0, 0)), processing_time
                            )
                            
                            # Display word
                            self.root.after(0, lambda w=word: self.detected_letter.set(f"[*] {w}"))
                            
                            if self.dev_console:
                                self.dev_console.add_log(f"Word detected: {word} (conf: {word_conf:.2f})")
                            
                            # Reset buffer and enter cooldown
                            self.alphabet_classifier.reset_dynamic_buffer()
                            self.detection_state = "COOLDOWN"
                            self.last_detection_time = current_time
                        else:
                            # Still buffering frames for word
                            buffer_size = len(self.alphabet_classifier.dynamic_gesture_buffer)
                            self.root.after(0, lambda: self.detected_letter.set(f"[REC] {buffer_size}"))
                    
                    # === STATIC GESTURE RECOGNITION (LETTERS) ===
                    else:
                        # Reset dynamic buffer when doing static detection
                        if hasattr(self.alphabet_classifier, 'reset_dynamic_buffer'):
                            self.alphabet_classifier.reset_dynamic_buffer()
                        
                        # Check cooldown state
                        if self.detection_state == "COOLDOWN":
                            if current_time - self.last_detection_time >= self.cooldown_duration:
                                self.detection_state = "IDLE"
                                self.stable_detection_frames = 0
                                if self.dev_console:
                                    self.dev_console.add_log("Cooldown ended, ready for new letter")
                            else:
                                # Still in cooldown - show cooldown indicator but continue processing
                                self.root.after(0, lambda: self.detected_letter.set("[WAIT]"))
                                # Skip recognition during cooldown
                                letter = None
                                conf = 0.0
                                is_stable = False
                        
                        # Recognize letter (only if NOT in cooldown)
                        if self.detection_state != "COOLDOWN":
                            letter, conf, is_stable = self.alphabet_classifier.recognize_with_stability(landmarks)
                        else:
                            letter = None
                            conf = 0.0
                        
                        if letter:
                            recognized_letter = letter
                            letter_confidence = conf
                            
                            # State transition: IDLE -> LISTENING
                            if self.detection_state == "IDLE":
                                self.detection_state = "LISTENING"
                            
                            # Check for stable detection
                            if self.detection_state == "LISTENING":
                                if letter == self.last_detected_letter:
                                    self.stable_detection_frames += 1
                                else:
                                    self.stable_detection_frames = 1
                                    self.last_detected_letter = letter
                                
                                # Update display
                                self.root.after(0, lambda l=letter: self.detected_letter.set(l))
                                
                                # Add to word buffer when stable
                                if self.stable_detection_frames >= self.min_stable_frames:
                                    current_word = self.current_word.get()
                                    self.current_word.set(current_word + letter)
                                    
                                    # Log detection
                                    self.session_logger.log_detection(
                                        self.frame_count, letter, conf, 
                                        current_word + letter, self.sentence_text.get(),
                                        hand_info.get('position', (0, 0)), processing_time
                                    )
                                    
                                    # Check for autocorrect suggestions
                                    if self.autocorrect.enabled:
                                        suggestions = self.autocorrect.get_suggestions(current_word + letter)
                                        if suggestions:
                                            self.correction_suggestion.set(f"Did you mean: {', '.join(suggestions[:2])}?")
                                        else:
                                            self.correction_suggestion.set("")
                                    
                                    if self.dev_console:
                                        self.dev_console.add_log(f"Letter added: {letter} (conf: {conf:.2f}, stable: {self.stable_detection_frames})")
                                    
                                    # Transition to COOLDOWN
                                    self.detection_state = "COOLDOWN"
                                    self.last_detection_time = current_time
                                    self.stable_detection_frames = 0
                                    self.last_detected_letter = None
                        else:
                            # No letter detected
                            if self.detection_state == "LISTENING":
                                self.stable_detection_frames = 0
                                self.last_detected_letter = None
                            self.root.after(0, lambda: self.detected_letter.set("--"))
                        
                except Exception as e:
                    if self.dev_console:
                        self.dev_console.add_log(f"Alphabet recognition error: {e}", "ERROR")
            else:
                # No hand detected - reset to IDLE and clear buffers
                if self.detection_state != "COOLDOWN":
                    self.detection_state = "IDLE"
                    self.stable_detection_frames = 0
                    self.last_detected_letter = None
                if hasattr(self, 'alphabet_classifier') and hasattr(self.alphabet_classifier, 'reset_dynamic_buffer'):
                    self.alphabet_classifier.reset_dynamic_buffer()
                if hasattr(self, 'prev_hand_landmarks'):
                    delattr(self, 'prev_hand_landmarks')
                self.root.after(0, lambda: self.detected_letter.set("--"))
            
            # Log hand detection event
            if landmarks is not None:
                self.session_logger.log_hand_detected()
            
            # Update developer console
            if self.dev_console and self.frame_count % 10 == 0:  # Update every 10 frames
                self.dev_console.update_stats(
                    frame_count=self.frame_count,
                    detection_count=self.detection_count,
                    fps=f"{fps:.1f}" if 'fps' in locals() else "0",
                    landmarks=hand_info.get('landmarks_count', 0),
                    handedness=hand_info.get('handedness', 'None'),
                    position=hand_info.get('position', (0, 0)),
                    gesture=gesture,
                    confidence=f"{confidence:.1f}%"
                )
                
                if gesture != "NOT DETECTED" and self.frame_count % 100 == 0:
                    self.dev_console.add_log(f"Hand detected: {gesture} ({confidence:.1f}%)")
            
            # Update detection confidence
            self.root.after(0, lambda c=confidence: self.detection_confidence.set(f"Confidence: {c:.1f}%"))
            
            # Update visual gesture display
            if gesture == "NOT DETECTED":
                self.visual_gestur.set("NOT DETECTED")
                self.root.after(0, lambda: self.visual_label.config(fg="#666666"))
            else:
                # Color based on position
                if gesture == "LEFT":
                    self.root.after(0, lambda: self.visual_label.config(fg="#00ffff"))
                elif gesture == "RIGHT":
                    self.root.after(0, lambda: self.visual_label.config(fg="#ff00ff"))
                else:  # CENTRE
                    self.root.after(0, lambda: self.visual_label.config(fg="#00ff00"))
                
            self.root.after(0, lambda g=gesture: self.visual_gestur.set(g))
            
            if gesture != "NOT DETECTED" and self.mqtt_connected and hasattr(self, 'client'):
                try:
                    self.client.publish(MQTT_TOPIC_VISUAL, gesture)
                except Exception as e:
                    print(f"[ERROR] MQTT publish failed: {e}")
                
            # No calibration box needed with MediaPipe
            
            # Convert and resize to FIXED dimensions to prevent expansion
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            
            # Use FIXED size (16:9 aspect ratio) - prevents expanding beyond left frame
            fixed_width = 800
            fixed_height = 600
            frame_rgb = cv2.resize(frame_rgb, (fixed_width, fixed_height), interpolation=cv2.INTER_LINEAR)
            
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Store reference and update via after_idle for smooth rendering
            self.image_queue = imgtk
            self.root.after_idle(self.update_video_frame)
        
        self.cap.release()
        print("Camera Shutdown")
        #---------------------------------------
    
    def update_video_frame(self):
        """Thread-safe method to update video frame in GUI"""
        if self.image_queue:
            self.video_label.imgtk = self.image_queue
            self.video_label.configure(image=self.image_queue)
    
    def on_closing(self):
        # Make shutdown idempotent
        if not self.running:
            return
            
        print("[INFO] Closing Application...")
        
        # Stop camera thread
        self.running = False
        
        # Close camera capture
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            print("[INFO] Camera released")
        
        # Destroy OpenCV windows
        cv2.destroyAllWindows()
        
        # Wait for video thread with reasonable timeout
        if hasattr(self, 'video_thread') and self.video_thread.is_alive():
            print("[INFO] Stopping camera thread...")
            self.video_thread.join(timeout=1.0)  # 1 second timeout
        
        # Disconnect MQTT (only once)
        if hasattr(self, 'client'):
            try:
                print("[INFO] Disconnecting MQTT...")
                self.client.loop_stop()
                self.client.disconnect()
            except Exception as e:
                print(f"[WARN] MQTT disconnect error: {e}")
        
        # Save session logs
        if hasattr(self, 'session_logger'):
            try:
                self.session_logger.close()
                print("[INFO] Session logs saved")
            except Exception as e:
                print(f"[WARN] Session logger error: {e}")
        
        # Close resources
        if hasattr(self, 'tracker'):
            try:
                self.tracker.close()
            except:
                pass
        
        if hasattr(self, 'alphabet_classifier'):
            try:
                self.alphabet_classifier.close()
            except:
                pass
        
        print("[INFO] Shutdown complete")
        self.root.destroy()
        
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Sign Language Reader - Gesture Recognition System")
    root.geometry("1200x700")
    root.minsize(1300, 800)
    root.configure(bg="#1a1a1a")
    app = gestureApp(root)
    root.mainloop()