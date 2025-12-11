"""
SIGN LANGUAGE READER - MAIN LAUNCHER
Unified entry point for all applications
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

# Import centralized path configuration
try:
    from path_config import (
        PROJECT_ROOT, MODEL_FILE, DATASET_FILE,
        BUILT_IN_DIR, ensure_directories
    )
except ImportError:
    # Fallback if path_config not available
    def get_base_path():
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))
    
    PROJECT_ROOT = get_base_path()
    BUILT_IN_DIR = os.path.join(PROJECT_ROOT, 'built-in')
    MODEL_FILE = os.path.join(PROJECT_ROOT, 'built-in', 'dataset', 'model.p')
    DATASET_FILE = os.path.join(PROJECT_ROOT, 'built-in', 'dataset', 'data.pickle')
    
    def ensure_directories():
        os.makedirs(os.path.join(PROJECT_ROOT, 'built-in', 'dataset'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data', 'static', 'alphabet'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data', 'static', 'numbers'), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data', 'dynamic', 'words'), exist_ok=True)

# Ensure directories exist
ensure_directories()

# Define executable paths (check both .exe and .py)
DESKTOP_APP_EXE = os.path.join(BUILT_IN_DIR, 'desktop_app.exe')
COLLECT_DATA_EXE = os.path.join(BUILT_IN_DIR, 'collect_data.exe')
TRAIN_MODEL_EXE = os.path.join(BUILT_IN_DIR, 'train_model.exe')

# Fallback to .py if .exe not found (for development)
DESKTOP_APP_FILE = DESKTOP_APP_EXE if os.path.exists(DESKTOP_APP_EXE) else os.path.join(PROJECT_ROOT, 'desktop_app.py')
COLLECT_DATA_FILE = COLLECT_DATA_EXE if os.path.exists(COLLECT_DATA_EXE) else os.path.join(PROJECT_ROOT, 'collect_data.py')
TRAIN_MODEL_FILE = TRAIN_MODEL_EXE if os.path.exists(TRAIN_MODEL_EXE) else os.path.join(PROJECT_ROOT, 'train_model.py')


class LauncherApp:
    """
    Main launcher application with dark theme
    - Check model/dataset availability
    - Launch appropriate applications
    - User-friendly error messages
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("SIGN LANGUAGE READER - Launcher")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1000 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f"1000x700+{x}+{y}")
        
        # Status variables
        self.model_exists = os.path.exists(MODEL_FILE)
        self.dataset_exists = os.path.exists(DATASET_FILE)
        
        self._build_ui()
        self._update_status()
        
        # Auto-refresh every 2 seconds to detect new dataset/model
        self._start_auto_refresh()
    
    def _build_ui(self):
        """Build the launcher interface"""
        # Main container
        main_container = tk.Frame(self.root, bg="#1a1a1a", padx=40, pady=40)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # ========== HEADER ==========
        header_frame = tk.Frame(main_container, bg="#1a1a1a")
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        title = tk.Label(header_frame, 
                        text="SIGN LANGUAGE READER",
                        font=("Helvetica", 28, "bold"),
                        fg="#00ff88", bg="#1a1a1a")
        title.pack()
        
        subtitle = tk.Label(header_frame,
                           text="Gesture Recognition System",
                           font=("Helvetica", 14),
                           fg="#888888", bg="#1a1a1a")
        subtitle.pack(pady=(5, 0))
        
        # ========== STATUS PANEL ==========
        status_frame = tk.Frame(main_container, bg="#2b2b2b", relief=tk.RAISED, bd=3)
        status_frame.pack(fill=tk.X, pady=(0, 30))
        
        status_inner = tk.Frame(status_frame, bg="#2b2b2b", padx=20, pady=15)
        status_inner.pack(fill=tk.BOTH)
        
        tk.Label(status_inner, text="SYSTEM STATUS",
                font=("Helvetica", 12, "bold"),
                fg="#00d4ff", bg="#2b2b2b").pack(anchor=tk.W)
        
        tk.Frame(status_inner, height=2, bg="#00d4ff").pack(fill=tk.X, pady=(5, 10))
        
        # Status items
        status_content = tk.Frame(status_inner, bg="#2b2b2b")
        status_content.pack(fill=tk.X)
        
        # Dataset status
        self.dataset_status_label = tk.Label(status_content,
                                             text="● Dataset: Checking...",
                                             font=("Courier", 11),
                                             fg="#888888", bg="#2b2b2b")
        self.dataset_status_label.pack(anchor=tk.W, pady=3)
        
        # Model status
        self.model_status_label = tk.Label(status_content,
                                           text="● Model: Checking...",
                                           font=("Courier", 11),
                                           fg="#888888", bg="#2b2b2b")
        self.model_status_label.pack(anchor=tk.W, pady=3)
        
        # ========== MAIN ACTIONS ==========
        actions_frame = tk.Frame(main_container, bg="#1a1a1a")
        actions_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Desktop App
        left_panel = tk.Frame(actions_frame, bg="#2b2b2b", relief=tk.RAISED, bd=3)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        left_inner = tk.Frame(left_panel, bg="#2b2b2b", padx=25, pady=25)
        left_inner.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(left_inner, text="RUN APPLICATION",
                font=("Helvetica", 14, "bold"),
                fg="#00ff88", bg="#2b2b2b").pack(pady=(0, 10))
        
        tk.Frame(left_inner, height=2, bg="#00ff88").pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(left_inner,
                text="Start the sign language\ndetection application",
                font=("Helvetica", 11),
                fg="#aaaaaa", bg="#2b2b2b",
                justify=tk.CENTER).pack(pady=(0, 30))
        
        self.run_app_btn = tk.Button(left_inner,
                                     text="▶ LAUNCH\nDESKTOP APP",
                                     font=("Helvetica", 16, "bold"),
                                     bg="#00ff00", fg="#000000",
                                     activebackground="#00dd00",
                                     command=self._launch_desktop_app,
                                     padx=30, pady=30,
                                     height=4)
        self.run_app_btn.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(left_inner,
                text="Requires trained model",
                font=("Helvetica", 9, "italic"),
                fg="#666666", bg="#2b2b2b").pack()
        
        # Right side: Data Creation
        right_panel = tk.Frame(actions_frame, bg="#2b2b2b", relief=tk.RAISED, bd=3)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        right_inner = tk.Frame(right_panel, bg="#2b2b2b", padx=25, pady=25)
        right_inner.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(right_inner, text="CREATE/TRAIN MODEL",
                font=("Helvetica", 14, "bold"),
                fg="#ff9500", bg="#2b2b2b").pack(pady=(0, 10))
        
        tk.Frame(right_inner, height=2, bg="#ff9500").pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(right_inner,
                text="Collect data and train\nthe recognition model",
                font=("Helvetica", 11),
                fg="#aaaaaa", bg="#2b2b2b",
                justify=tk.CENTER).pack(pady=(0, 20))
        
        # Data collection button
        self.collect_btn = tk.Button(right_inner,
                                     text="📷 COLLECT DATA",
                                     font=("Helvetica", 13, "bold"),
                                     bg="#0088ff", fg="#ffffff",
                                     activebackground="#0066cc",
                                     command=self._launch_collect_data,
                                     padx=20, pady=15)
        self.collect_btn.pack(fill=tk.X, pady=(0, 15))
        
        # Training button
        self.train_btn = tk.Button(right_inner,
                                   text="🎓 TRAIN MODEL",
                                   font=("Helvetica", 13, "bold"),
                                   bg="#666666", fg="#aaaaaa",
                                   state=tk.DISABLED,
                                   command=self._launch_train_model,
                                   padx=20, pady=15)
        self.train_btn.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(right_inner,
                text="Collect data first,\nthen train the model",
                font=("Helvetica", 9, "italic"),
                fg="#666666", bg="#2b2b2b",
                justify=tk.CENTER).pack()
        
        # ========== FOOTER ==========
        footer = tk.Frame(main_container, bg="#1a1a1a")
        footer.pack(fill=tk.X, pady=(20, 0))
        
        tk.Button(footer, text="Refresh Status",
                 font=("Helvetica", 10),
                 bg="#444444", fg="#ffffff",
                 command=self._refresh_status,
                 padx=20, pady=8).pack(side=tk.LEFT)
        
        tk.Button(footer, text="Exit",
                 font=("Helvetica", 10),
                 bg="#664444", fg="#ffffff",
                 command=self.root.destroy,
                 padx=30, pady=8).pack(side=tk.RIGHT)
    
    def _update_status(self):
        """Update status indicators"""
        # Check files
        self.model_exists = os.path.exists(MODEL_FILE)
        self.dataset_exists = os.path.exists(DATASET_FILE)
        
        # Update dataset status
        if self.dataset_exists:
            dataset_size = os.path.getsize(DATASET_FILE) / 1024
            self.dataset_status_label.config(
                text=f"● Dataset: Ready ({dataset_size:.1f} KB)",
                fg="#00ff00"
            )
            # Enable training button
            self.train_btn.config(state=tk.NORMAL, bg="#ffaa00", fg="#000000")
        else:
            self.dataset_status_label.config(
                text="● Dataset: Not Found",
                fg="#ff6600"
            )
            self.train_btn.config(state=tk.DISABLED, bg="#666666", fg="#aaaaaa")
        
        # Update model status
        if self.model_exists:
            model_size = os.path.getsize(MODEL_FILE) / 1024
            self.model_status_label.config(
                text=f"● Model: Ready ({model_size:.1f} KB)",
                fg="#00ff00"
            )
            # Enable run button
            self.run_app_btn.config(state=tk.NORMAL, bg="#00ff00", fg="#000000")
        else:
            self.model_status_label.config(
                text="● Model: Not Found",
                fg="#ff0000"
            )
            self.run_app_btn.config(state=tk.NORMAL, bg="#ff6600", fg="#ffffff")
    
    def _refresh_status(self):
        """Refresh status display"""
        self._update_status()
        messagebox.showinfo("Status Refreshed", 
                           "System status has been updated.",
                           parent=self.root)
    
    def _start_auto_refresh(self):
        """Start automatic status refresh every 2 seconds"""
        self._update_status()
        # Schedule next refresh
        self.root.after(2000, self._start_auto_refresh)
    
    def _launch_desktop_app(self):
        """Launch main desktop application"""
        if not self.model_exists:
            response = messagebox.askyesno(
                "Model Not Found",
                "The trained model is not available.\n\n"
                "You need to:\n"
                "1. Collect training data\n"
                "2. Train the model\n\n"
                "Do you want to create the model now?",
                icon='warning',
                parent=self.root
            )
            
            if response:
                self._launch_collect_data()
            return
        
        # Launch desktop app
        try:
            if DESKTOP_APP_FILE.endswith('.exe'):
                # Launch .exe directly
                subprocess.Popen([DESKTOP_APP_FILE])
            else:
                # Launch .py with Python
                subprocess.Popen([sys.executable, DESKTOP_APP_FILE])
            # App launched - no notification needed
        except Exception as e:
            messagebox.showerror("Launch Error",
                                f"Failed to launch desktop app:\n{str(e)}",
                                parent=self.root)
    
    def _launch_collect_data(self):
        """Launch data collection app"""
        if not os.path.exists(COLLECT_DATA_FILE):
            messagebox.showerror("File Not Found",
                                f"collect_data not found at:\n{COLLECT_DATA_FILE}",
                                parent=self.root)
            return
        
        try:
            if COLLECT_DATA_FILE.endswith('.exe'):
                # Launch .exe directly
                subprocess.Popen([COLLECT_DATA_FILE])
            else:
                # Launch .py with Python
                subprocess.Popen([sys.executable, COLLECT_DATA_FILE])
            # App launched - no notification needed
        except Exception as e:
            messagebox.showerror("Launch Error",
                                f"Failed to launch data collection:\n{str(e)}",
                                parent=self.root)
    
    def _launch_train_model(self):
        """Launch model training app"""
        if not self.dataset_exists:
            messagebox.showwarning("Dataset Required",
                                  "No dataset found!\n\n"
                                  "Please collect training data first using\n"
                                  "the 'COLLECT DATA' button.",
                                  parent=self.root)
            return
        
        if not os.path.exists(TRAIN_MODEL_FILE):
            messagebox.showerror("File Not Found",
                                f"train_model not found at:\n{TRAIN_MODEL_FILE}",
                                parent=self.root)
            return
        
        try:
            if TRAIN_MODEL_FILE.endswith('.exe'):
                # Launch .exe directly
                subprocess.Popen([TRAIN_MODEL_FILE])
            else:
                # Launch .py with Python
                subprocess.Popen([sys.executable, TRAIN_MODEL_FILE])
            # App launched - no notification needed
        except Exception as e:
            messagebox.showerror("Launch Error",
                                f"Failed to launch model training:\n{str(e)}",
                                parent=self.root)


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()
