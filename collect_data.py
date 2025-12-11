"""
GUI Sign Language Data Collection System (Tkinter/OpenCV/MediaPipe)
MENGGANTIKAN collect_data.py LAMA.
FINAL: Menggunakan styling ttk standar (konsisten dengan desktop_app.py).
"""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import sys

# Tambahkan path built-in
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'built-in'))

from PIL import Image, ImageTk
import cv2
import mediapipe as mp 
import warnings
import logging

# --- FULL CODE Suppress ALL warnings and logs ---
if sys.platform == 'win32':
    try:
        stderr_fd = sys.stderr.fileno()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        sys.stderr = open(os.devnull, 'w')
    except Exception:
        pass 
        
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
# ----------------------------------------------------

# --- Configuration & MediaPipe Initialization ---
# Use path_config - NO FALLBACK, direct sys.executable location
from path_config import PROJECT_ROOT, STATIC_ALPHABET_DIR, STATIC_NUMBERS_DIR, DYNAMIC_WORDS_DIR
BASE_DIR = PROJECT_ROOT
DATA_STRUCTURE = {
    'static': {
        'alphabet': STATIC_ALPHABET_DIR,
        'numbers': STATIC_NUMBERS_DIR
    },
    'dynamic': {
        'words': DYNAMIC_WORDS_DIR
    }
}
print(f"[PATH CONFIG] BASE_DIR: {BASE_DIR}")
print(f"[PATH CONFIG] ALPHABET: {STATIC_ALPHABET_DIR}")
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 


class DataCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DATA COLLECTION - Sign Language Reader")
        self.root.geometry("1300x800")
        self.root.configure(bg="#1a1a1a")
        
        # --- Theme Initialization (Match desktop_app.py) ---
        self.style = ttk.Style(root)
        self.style.theme_use('clam')
        # ----------------------------------------------------
        
        self.is_video = tk.BooleanVar(value=False)
        self.item_list = tk.StringVar(value="")
        self.dataset_size = tk.IntVar(value=150)
        self.status_msg = tk.StringVar(value="Ready. Configure settings.")
        self.current_item = tk.StringVar(value="N/A")
        self.progress_count = tk.IntVar(value=0)
        
        self.cap = None
        self.running = False
        self.collecting = False
        self.current_save_path = ""
        self.frames_to_save = []
        
        # FPS control (fix flickering)
        self.fps_limit = 30
        self.frame_delay = 1.0 / self.fps_limit
        self.last_frame_time = 0
        self.image_queue = None
        
        # Log window
        self.log_window = None
        self.log_text_widget = None
        
        self._setup_ui()
        self._initialize_camera()

    def _setup_ui(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ========== LEFT SIDE: VIDEO FEED ==========
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display container
        video_container = ttk.Frame(left_frame, relief=tk.RIDGE, borderwidth=3)
        video_container.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_container, 
                                     text="[CAMERA] Initializing...\n\nPlease wait...",
                                     font=("Helvetica", 12), 
                                     anchor=tk.CENTER,
                                     background="black",
                                     foreground="white")
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # ========== RIGHT SIDE: CONTROLS (Match desktop_app.py style) ==========
        right_frame = tk.Frame(main_frame, width=380, bg="#1a1a1a")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        # Scrollable container
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
        
        # Inner frame
        inner_frame = tk.Frame(scrollable_frame, bg="#1a1a1a", padx=10, pady=10)
        inner_frame.pack(fill=tk.BOTH, expand=True)

        # ========== SECTION 1: SETTINGS ==========
        settings_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        settings_section.pack(fill=tk.X, pady=(0, 10))
        
        settings_inner = tk.Frame(settings_section, bg="#2b2b2b", padx=8, pady=8)
        settings_inner.pack(fill=tk.BOTH)
        
        settings_header = tk.Label(settings_inner, text="[SETTINGS]", 
                                   font=("Helvetica", 10, "bold"), 
                                   fg="#00ff88", bg="#2b2b2b")
        settings_header.pack(anchor=tk.W)
        
        tk.Frame(settings_inner, height=1, bg="#00ff88").pack(fill=tk.X, pady=(2, 6))
        
        # Data Type
        tk.Label(settings_inner, text="Data Type:", 
                font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b").pack(anchor=tk.W)
        
        type_frame = tk.Frame(settings_inner, bg="#2b2b2b")
        type_frame.pack(fill=tk.X, pady=(0, 6))
        
        self.static_btn = tk.Button(type_frame, text="Static (Images)", 
                                    font=("Helvetica", 9, "bold"),
                                    bg="#00ff00", fg="#000000",
                                    command=lambda: self._update_data_type(False),
                                    padx=10, pady=5)
        self.static_btn.pack(side=tk.LEFT, padx=(0, 3))
        
        self.dynamic_btn = tk.Button(type_frame, text="Dynamic (Video)", 
                                     font=("Helvetica", 9, "bold"),
                                     bg="#444444", fg="#ffffff",
                                     command=lambda: self._update_data_type(True),
                                     padx=10, pady=5)
        self.dynamic_btn.pack(side=tk.LEFT)
        
        # Items input
        tk.Label(settings_inner, text="Items (e.g., A,B,C):", 
                font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b").pack(anchor=tk.W, pady=(6, 0))
        
        self.item_entry = tk.Entry(settings_inner, textvariable=self.item_list,
                                   font=("Courier", 12, "bold"), 
                                   bg="#1a1a1a", fg="#ffff00",
                                   insertbackground="#ffff00")
        self.item_entry.pack(fill=tk.X, pady=(0, 6), ipady=4)
        
        # Sample count
        tk.Label(settings_inner, text="Samples per Item:", 
                font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b").pack(anchor=tk.W)
        
        self.size_entry = tk.Entry(settings_inner, textvariable=self.dataset_size,
                                   font=("Courier", 12, "bold"),
                                   bg="#1a1a1a", fg="#00ffff",
                                   insertbackground="#00ffff")
        self.size_entry.pack(fill=tk.X, ipady=4)

        # ========== SECTION 2: COLLECTION STATUS ==========
        status_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        status_section.pack(fill=tk.X, pady=(0, 10))
        
        status_inner = tk.Frame(status_section, bg="#2b2b2b", padx=8, pady=8)
        status_inner.pack(fill=tk.BOTH)
        
        status_header = tk.Label(status_inner, text="[COLLECTION STATUS]", 
                                font=("Helvetica", 10, "bold"), 
                                fg="#ff9500", bg="#2b2b2b")
        status_header.pack(anchor=tk.W)
        
        tk.Frame(status_inner, height=1, bg="#ff9500").pack(fill=tk.X, pady=(2, 6))
        
        # Current Item
        tk.Label(status_inner, text="Current Item:", 
                font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b").pack(anchor=tk.W)
        
        tk.Label(status_inner, textvariable=self.current_item, 
                font=("Helvetica", 24, "bold"), fg="#00ffff", bg="#2b2b2b").pack(anchor=tk.W, pady=(0, 6))
        
        # Progress bar
        tk.Label(status_inner, text="Progress:", 
                font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b").pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(status_inner, orient="horizontal", 
                                           length=300, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))
        
        # Status message
        tk.Label(status_inner, textvariable=self.status_msg, 
                font=("Helvetica", 8), fg="#ffff00", bg="#2b2b2b",
                wraplength=320, justify=tk.LEFT).pack(anchor=tk.W)

        # ========== SECTION 3: CONTROL BUTTONS ==========
        control_section = tk.Frame(inner_frame, bg="#2b2b2b", relief=tk.RAISED, borderwidth=2)
        control_section.pack(fill=tk.X, pady=(0, 10))
        
        control_inner = tk.Frame(control_section, bg="#2b2b2b", padx=8, pady=8)
        control_inner.pack(fill=tk.BOTH)
        
        self.start_btn = tk.Button(control_inner, text="START COLLECTION", 
                                   font=("Helvetica", 11, "bold"),
                                   bg="#00ff00", fg="#000000",
                                   activebackground="#00dd00",
                                   command=self._start_stop_collection,
                                   padx=20, pady=10)
        self.start_btn.pack(fill=tk.X, pady=(0, 6))
        
        # Info box
        info_text = ("Tips for Quality:\n"
                    "• Use good lighting\n"
                    "• Clear hand gesture\n"
                    "• Vary angles & distances\n"
                    "• Keep hand in frame\n")
        
        tk.Label(control_inner, text=info_text, 
                font=("Helvetica", 8), fg="#aaaaaa", bg="#2b2b2b",
                justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        
        # Exit button
        tk.Button(inner_frame, text="Exit", 
                 font=("Helvetica", 10, "bold"),
                 bg="#666666", fg="#ffffff",
                 command=self._on_closing,
                 padx=30, pady=8).pack(pady=10)
        
        # View Logs button
        tk.Button(inner_frame, text="View Logs", 
                 font=("Helvetica", 10, "bold"),
                 bg="#0088ff", fg="#ffffff",
                 command=self._show_log_window,
                 padx=30, pady=8).pack(pady=(0, 10))
        
    def _update_data_type(self, is_dynamic):
        """Update data type between static (images) and dynamic (video)"""
        self.is_video.set(is_dynamic)
        self.dataset_size.set(20 if is_dynamic else 150)
        
        # Update button styling
        if is_dynamic:
            self.static_btn.config(bg="#444444", fg="#ffffff")
            self.dynamic_btn.config(bg="#00ff00", fg="#000000")
        else:
            self.static_btn.config(bg="#00ff00", fg="#000000")
            self.dynamic_btn.config(bg="#444444", fg="#ffffff")
        
        self.status_msg.set("Set items and size, then click START.")
        
    def _initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_msg.set("ERROR: Cannot open camera.")
            self._log("Failed to open camera", "ERROR")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._log("Camera initialized successfully (640x480)", "SUCCESS")

    def _start_stop_collection(self):
        if not self.collecting:
            items = [i.strip().upper() for i in self.item_list.get().split(',') if i.strip()]
            if not items:
                messagebox.showerror("Error", "Please enter items to collect (e.g., A,B,C).")
                return

            self.items_queue = items
            self.current_item_index = 0
            self.current_sample_count = 0
            self.collecting = True
            self.start_btn.config(text="PAUSE COLLECTION", bg="#ff6600", fg="#ffffff")
            self.status_msg.set(f"Starting collection for: {self.items_queue[0]}")
            self._log(f"Started collection for items: {', '.join(items)}", "INFO")
            self._prepare_item_dir()
        else:
            self.collecting = False
            self.start_btn.config(text="RESUME COLLECTION", bg="#00dd00", fg="#000000")
            self.status_msg.set("Collection Paused.")
            self._log(f"Collection paused at {self.current_sample_count}/{self.dataset_size.get()} samples", "WARN")

    def _prepare_item_dir(self):
        item = self.items_queue[self.current_item_index]
        self.current_item.set(item)
        
        if self.is_video.get():
            category = 'words'
            item_path = os.path.join(DATA_STRUCTURE['dynamic'][category], item.lower())
        elif item.isdigit():
            category = 'numbers'
            item_path = os.path.join(DATA_STRUCTURE['static'][category], item.upper())
        else:
            category = 'alphabet'
            item_path = os.path.join(DATA_STRUCTURE['static'][category], item.upper())
            
        os.makedirs(item_path, exist_ok=True)
        self.current_save_path = item_path
        self.progress_bar.config(maximum=self.dataset_size.get(), value=self.current_sample_count)
        self.status_msg.set(f"Ready to collect {item}. Samples: {self.current_sample_count}/{self.dataset_size.get()}")
        self._log(f"Prepared directory for '{item}': {item_path}", "INFO")


    def _video_loop(self):
        while self.running:
            # FPS limiting to reduce flickering
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_delay:
                time.sleep(0.001)
                continue
            
            ret, frame = self.cap.read()
            if not ret: 
                break
            
            self.last_frame_time = current_time
            frame = cv2.flip(frame, 1)

            frame_to_save = frame.copy() 
            hand_results = None          
            hand_detected = False

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Hand Detection
            hand_results = hands_detector.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                hand_detected = True
                self._draw_feedback(frame, hand_results)

            # 2. Face Blurring
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    ih, iw, _ = frame.shape
                    bbox_c = detection.location_data.relative_bounding_box
                    face_x = max(0, int(bbox_c.xmin * iw))
                    face_y = max(0, int(bbox_c.ymin * ih))
                    face_w = min(iw - face_x, int(bbox_c.width * iw))
                    face_h = min(ih - face_y, int(bbox_c.height * ih))
                    
                    hand_in_face = False 

                    if face_w > 0 and face_h > 0 and not hand_in_face:
                        face_region = frame_to_save[face_y:face_y+face_h, face_x:face_x+face_w]
                        face_region = cv2.GaussianBlur(face_region, (99, 99), 30) 
                        frame_to_save[face_y:face_y+face_h, face_x:face_x+face_w] = face_region
            
            # 3. Collection/Saving Logic
            if self.collecting and hand_detected:
                if self.is_video.get():
                    self.frames_to_save.append(frame_to_save)
                    self.status_msg.set(f"RECORDING... ({len(self.frames_to_save)} frames)")
                else:
                    if self.current_sample_count < self.dataset_size.get():
                        self._save_static_frame(frame_to_save)
            
            # Use image queue pattern to prevent flickering
            img_tk = self._cv2_to_tk(frame) 
            self.image_queue = img_tk
            self.root.after_idle(self._update_video_frame) 

    # --- Utility Functions ---
    def _update_video_frame(self):
        """Update video frame using after_idle pattern to prevent flickering"""
        if self.image_queue:
            self.video_label.imgtk = self.image_queue
            self.video_label.config(image=self.image_queue)
    
    def _draw_feedback(self, frame, hand_results):
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

    def _cv2_to_tk(self, frame):
        try:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480), Image.LANCZOS) 
            imgtk = ImageTk.PhotoImage(image=img)
            return imgtk
        except NameError:
            return None 

    def _on_closing(self):
        """Handle window close - create dataset if data was collected"""
        self.running = False
        if self.cap: 
            self.cap.release()
        cv2.destroyAllWindows()
        if self.log_window:
            self.log_window.destroy()
        
        # Check if any data was collected
        data_collected = False
        for category_type in DATA_STRUCTURE.values():
            for category_path in category_type.values():
                if os.path.exists(category_path):
                    # Check if folder has any images
                    for item_folder in os.listdir(category_path):
                        item_path = os.path.join(category_path, item_folder)
                        if os.path.isdir(item_path):
                            images = [f for f in os.listdir(item_path) if f.endswith('.jpg')]
                            if len(images) > 0:
                                data_collected = True
                                break
                if data_collected:
                    break
            if data_collected:
                break
        
        # If data was collected, create dataset
        if data_collected:
            response = messagebox.askyesno(
                "Create Dataset",
                "Data collection complete!\n\n"
                "Would you like to create the training dataset now?\n"
                "(This will process all collected images)",
                parent=self.root
            )
            
            if response:
                # Create log window for dataset creation
                log_window = tk.Toplevel(self.root)
                log_window.title("Creating Dataset")
                log_window.geometry("700x500")
                log_window.configure(bg="#1a1a1a")
                
                # Header
                header = tk.Frame(log_window, bg="#1a1a1a", pady=10)
                header.pack(fill=tk.X)
                tk.Label(header,
                        text="CREATING DATASET FROM IMAGES",
                        font=("Helvetica", 12, "bold"),
                        fg="#00ff88", bg="#1a1a1a").pack()
                
                # Log text area
                log_frame = tk.Frame(log_window, bg="#0a0a0a", relief=tk.SUNKEN, bd=2)
                log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
                
                log_text = tk.Text(log_frame,
                                  bg="#0a0a0a", fg="#00ff00",
                                  font=("Consolas", 9),
                                  wrap=tk.WORD,
                                  state=tk.DISABLED)
                log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
                
                scrollbar = tk.Scrollbar(log_text, command=log_text.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                log_text.config(yscrollcommand=scrollbar.set)
                
                def log_message(msg, color="#00ff00"):
                    log_text.config(state=tk.NORMAL)
                    log_text.insert(tk.END, msg + "\n", (color,))
                    log_text.tag_config(color, foreground=color)
                    log_text.see(tk.END)
                    log_text.config(state=tk.DISABLED)
                    log_window.update()
                
                log_message("Starting dataset creation...", "#00ff88")
                log_message("This will process all collected images.", "#aaaaaa")
                log_message("="*60, "#555555")
                
                try:
                    # Re-enable stdout/stderr temporarily for exec
                    import sys
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    
                    # Capture output
                    import io
                    output_buffer = io.StringIO()
                    
                    class TeeOutput:
                        def __init__(self, log_func):
                            self.log_func = log_func
                            self.buffer = io.StringIO()
                        
                        def write(self, text):
                            if text.strip():
                                self.log_func(text.rstrip(), "#00ff00")
                            self.buffer.write(text)
                        
                        def flush(self):
                            pass
                    
                    tee_out = TeeOutput(log_message)
                    sys.stdout = tee_out
                    sys.stderr = tee_out
                    
                    # Debug: Show paths being used
                    log_message(f"[DEBUG] BASE_DIR: {BASE_DIR}", "#ffaa00")
                    log_message(f"[DEBUG] PROJECT_ROOT from path_config: {PROJECT_ROOT}", "#ffaa00")
                    
                    # Import create_dataset script
                    sys.path.insert(0, os.path.join(BASE_DIR, 'built-in'))
                    
                    # Read and execute create_dataset.py
                    create_dataset_path = os.path.join(BASE_DIR, 'built-in', 'create_dataset.py')
                    log_message(f"[DEBUG] Looking for create_dataset.py at: {create_dataset_path}", "#ffaa00")
                    if os.path.exists(create_dataset_path):
                        log_message(f"[DEBUG] create_dataset.py found! Reading...", "#00ff88")
                        with open(create_dataset_path, 'r', encoding='utf-8') as f:
                            script_code = f.read()
                        
                        log_message("="*60, "#555555")
                        log_message("Starting dataset creation...", "#00ff88")
                        log_message("="*60, "#555555")
                        
                        # Execute in current process
                        script_globals = {'__name__': '__main__', '__file__': create_dataset_path}
                        try:
                            exec(script_code, script_globals)
                        except SystemExit:
                            pass  # create_dataset calls sys.exit()
                        
                        # Restore stdout/stderr
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        
                        log_message("="*60, "#555555")
                        log_message("✓ Dataset created successfully!", "#00ff88")
                        log_message("You can now close this window and train the model.", "#aaaaaa")
                        
                        # Close button
                        close_btn = tk.Button(log_window,
                                            text="CLOSE",
                                            font=("Helvetica", 10, "bold"),
                                            bg="#00ff88", fg="#000000",
                                            command=log_window.destroy,
                                            padx=30, pady=10)
                        close_btn.pack(pady=10)
                    else:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        log_message(f"ERROR: create_dataset.py not found at:", "#ff0000")
                        log_message(create_dataset_path, "#ff0000")
                except Exception as e:
                    # Restore stdout/stderr
                    sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
                    sys.stderr = original_stderr if 'original_stderr' in locals() else sys.stderr
                    
                    log_message("="*60, "#555555")
                    log_message(f"ERROR: {str(e)}", "#ff0000")
                    log_message("Dataset creation failed.", "#ff0000")
        
        self.root.destroy()
    
    def _show_log_window(self):
        """Show or focus the log window"""
        if self.log_window is None or not self.log_window.winfo_exists():
            self._create_log_window()
        else:
            self.log_window.lift()
            self.log_window.focus_force()
    
    def _create_log_window(self):
        """Create a separate window for logs"""
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("Collection Logs - Sign Language Reader")
        self.log_window.geometry("700x500")
        self.log_window.configure(bg="#1a1a1a")
        
        # Main frame
        main_frame = tk.Frame(self.log_window, bg="#1a1a1a", padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Label(main_frame, text="[COLLECTION LOGS]", 
                         font=("Helvetica", 12, "bold"), 
                         fg="#00ff88", bg="#1a1a1a")
        header.pack(anchor=tk.W, pady=(0, 5))
        
        tk.Frame(main_frame, height=2, bg="#00ff88").pack(fill=tk.X, pady=(0, 10))
        
        # Log text area
        log_container = tk.Frame(main_frame, bg="#0a0a0a", relief=tk.SUNKEN, borderwidth=2)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text_widget = tk.Text(log_container, 
                                       bg="#0a0a0a", fg="#00ff00",
                                       font=("Courier", 9),
                                       wrap="word",
                                       yscrollcommand=scrollbar.set,
                                       state=tk.DISABLED)
        self.log_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar.config(command=self.log_text_widget.yview)
        
        # Configure tags for different log types
        self.log_text_widget.tag_config("INFO", foreground="#00ffff")
        self.log_text_widget.tag_config("SUCCESS", foreground="#00ff00")
        self.log_text_widget.tag_config("WARN", foreground="#ffaa00")
        self.log_text_widget.tag_config("ERROR", foreground="#ff0000")
        self.log_text_widget.tag_config("CAPTURE", foreground="#00ff88")
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg="#1a1a1a")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Button(button_frame, text="Clear Logs", 
                 font=("Helvetica", 9, "bold"),
                 bg="#ff6600", fg="#ffffff",
                 command=self._clear_logs,
                 padx=15, pady=5).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(button_frame, text="Close", 
                 font=("Helvetica", 9, "bold"),
                 bg="#666666", fg="#ffffff",
                 command=self.log_window.destroy,
                 padx=15, pady=5).pack(side=tk.LEFT)
        
        # Add initial log
        self._log("Log window opened", "INFO")
    
    def _clear_logs(self):
        """Clear all logs in the log window"""
        if self.log_text_widget:
            self.log_text_widget.config(state=tk.NORMAL)
            self.log_text_widget.delete(1.0, tk.END)
            self.log_text_widget.config(state=tk.DISABLED)
            self._log("Logs cleared", "INFO")
    
    def _log(self, message, level="INFO"):
        """Add a log entry to the log window"""
        if self.log_text_widget and self.log_text_widget.winfo_exists():
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            
            self.log_text_widget.config(state=tk.NORMAL)
            self.log_text_widget.insert(tk.END, log_entry, level)
            self.log_text_widget.see(tk.END)
            self.log_text_widget.config(state=tk.DISABLED)
        
    def _save_static_frame(self, frame_to_save):
        timestamp = int(time.time() * 1000)
        img_path = os.path.join(self.current_save_path, f'{timestamp}.jpg')
        
        try:
            cv2.imwrite(img_path, frame_to_save)
            
            # Verify file was actually saved
            if os.path.exists(img_path):
                file_size = os.path.getsize(img_path)
                self.current_sample_count += 1
                self.progress_bar.config(value=self.current_sample_count)
                self.status_msg.set(f"Captured: {self.current_sample_count}/{self.dataset_size.get()}")
                self._log(f"Saved image {self.current_sample_count}/{self.dataset_size.get()} → {img_path} ({file_size} bytes)", "CAPTURE")
                
                if self.current_sample_count >= self.dataset_size.get():
                    self._finish_current_item()
            else:
                error_msg = f"FAILED to save image to {img_path} - File not found after write"
                self.status_msg.set(f"ERROR: Save failed ({self.current_sample_count}/{self.dataset_size.get()})")
                self._log(error_msg, "ERROR")
                messagebox.showerror("Save Error", error_msg)
        except Exception as e:
            error_msg = f"FAILED to save image: {str(e)}"
            self.status_msg.set(f"ERROR: {str(e)}")
            self._log(f"{error_msg} → Path: {img_path}", "ERROR")
            messagebox.showerror("Save Error", error_msg)
            
    def _finish_current_item(self):
        item_name = self.current_item.get()
        
        # Verify all files were saved
        if os.path.exists(self.current_save_path):
            saved_files = [f for f in os.listdir(self.current_save_path) if f.endswith('.jpg')]
            total_files = len(saved_files)
            
            if total_files >= self.current_sample_count:
                messagebox.showinfo("Complete", f"Finished collecting {item_name}\n\nSaved {total_files} images to:\n{self.current_save_path}")
                self._log(f"Completed collection for '{item_name}': {total_files} files saved", "SUCCESS")
                self._log(f"Storage path: {self.current_save_path}", "INFO")
            else:
                error_msg = f"WARNING: Expected {self.current_sample_count} files but found only {total_files}"
                messagebox.showwarning("Incomplete Save", f"{error_msg}\n\nPath: {self.current_save_path}")
                self._log(error_msg, "WARN")
                self._log(f"Check directory: {self.current_save_path}", "WARN")
        else:
            error_msg = f"CRITICAL ERROR: Save directory not found: {self.current_save_path}"
            messagebox.showerror("Directory Error", error_msg)
            self._log(error_msg, "ERROR")
        
        self.current_item_index += 1
        self.current_sample_count = 0
        self.frames_to_save = []
        
        if self.current_item_index < len(self.items_queue):
            self._prepare_item_dir()
        else:
            self.collecting = False
            self.status_msg.set("ALL COLLECTION COMPLETE. Ready for Training.")
            self.start_btn.config(text="START COLLECTION", bg="#00ff00", fg="#000000")
            self._log("All items collection completed successfully!", "SUCCESS")

if __name__ == '__main__':
    root = tk.Tk()
    app = DataCollectorGUI(root)
    root.mainloop()