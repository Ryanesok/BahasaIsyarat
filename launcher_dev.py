"""
DEVELOPMENT MODE LAUNCHER
Run Python files directly WITHOUT building .exe
Use this for FAST TESTING before final build
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

# Force development mode - use .py files
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILT_IN_DIR = os.path.join(PROJECT_ROOT, 'built-in')
DATASET_FILE = os.path.join(BUILT_IN_DIR, 'dataset', 'data.pickle')
MODEL_FILE = os.path.join(BUILT_IN_DIR, 'dataset', 'model.p')

# Always use .py files in dev mode
DESKTOP_APP_FILE = os.path.join(PROJECT_ROOT, 'desktop_app.py')
COLLECT_DATA_FILE = os.path.join(PROJECT_ROOT, 'collect_data.py')
TRAIN_MODEL_FILE = os.path.join(PROJECT_ROOT, 'train_model.py')

# Ensure directories
os.makedirs(os.path.join(BUILT_IN_DIR, 'dataset'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data', 'static', 'alphabet'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data', 'static', 'numbers'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'sign-language-detector-python', 'data', 'dynamic', 'words'), exist_ok=True)


class LauncherDevApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛠️ DEVELOPMENT MODE - Sign Language Reader")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)
        
        self.model_exists = os.path.exists(MODEL_FILE)
        self.dataset_exists = os.path.exists(DATASET_FILE)
        
        self._build_ui()
        self._update_status()
        self._start_auto_refresh()
    
    def _build_ui(self):
        main_container = tk.Frame(self.root, bg="#1a1a1a", padx=40, pady=40)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_container, bg="#1a1a1a")
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        title = tk.Label(header_frame, 
                        text="🛠️ DEVELOPMENT MODE",
                        font=("Helvetica", 24, "bold"),
                        fg="#ff9500", bg="#1a1a1a")
        title.pack()
        
        subtitle = tk.Label(header_frame,
                           text="Testing WITHOUT building .exe - Changes take effect IMMEDIATELY",
                           font=("Helvetica", 11),
                           fg="#00ff88", bg="#1a1a1a")
        subtitle.pack(pady=(5, 0))
        
        # Status panel
        status_frame = tk.Frame(main_container, bg="#2b2b2b", relief=tk.RAISED, bd=3)
        status_frame.pack(fill=tk.X, pady=(0, 30))
        
        status_inner = tk.Frame(status_frame, bg="#2b2b2b", padx=20, pady=15)
        status_inner.pack(fill=tk.BOTH)
        
        tk.Label(status_inner, text="SYSTEM STATUS",
                font=("Helvetica", 12, "bold"),
                fg="#00ff88", bg="#2b2b2b").pack(anchor=tk.W)
        
        tk.Frame(status_inner, height=2, bg="#00ff88").pack(fill=tk.X, pady=(5, 15))
        
        self.dataset_status_label = tk.Label(status_inner,
                                             text="● Dataset: Checking...",
                                             font=("Helvetica", 11),
                                             fg="#ffff00", bg="#2b2b2b")
        self.dataset_status_label.pack(anchor=tk.W, pady=3)
        
        self.model_status_label = tk.Label(status_inner,
                                           text="● Model: Checking...",
                                           font=("Helvetica", 11),
                                           fg="#ffff00", bg="#2b2b2b")
        self.model_status_label.pack(anchor=tk.W, pady=3)
        
        # Action buttons
        actions_frame = tk.Frame(main_container, bg="#1a1a1a")
        actions_frame.pack(fill=tk.BOTH, expand=True)
        
        # Desktop App button
        left_panel = tk.Frame(actions_frame, bg="#2b2b2b", relief=tk.RAISED, bd=3)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        left_inner = tk.Frame(left_panel, bg="#2b2b2b", padx=25, pady=25)
        left_inner.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(left_inner, text="RUN APPLICATION",
                font=("Helvetica", 14, "bold"),
                fg="#00ff88", bg="#2b2b2b").pack(pady=(0, 10))
        
        self.run_app_btn = tk.Button(left_inner,
                                     text="▶ LAUNCH\nDESKTOP APP",
                                     font=("Helvetica", 16, "bold"),
                                     bg="#00ff00", fg="#000000",
                                     activebackground="#00dd00",
                                     command=self._launch_desktop_app,
                                     padx=30, pady=30,
                                     height=4)
        self.run_app_btn.pack(fill=tk.X, pady=(0, 20))
        
        # Right panel
        right_panel = tk.Frame(actions_frame, bg="#2b2b2b", relief=tk.RAISED, bd=3)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        right_inner = tk.Frame(right_panel, bg="#2b2b2b", padx=25, pady=25)
        right_inner.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(right_inner, text="CREATE/TRAIN MODEL",
                font=("Helvetica", 14, "bold"),
                fg="#ff9500", bg="#2b2b2b").pack(pady=(0, 10))
        
        self.collect_btn = tk.Button(right_inner,
                                     text="📷 COLLECT DATA",
                                     font=("Helvetica", 13, "bold"),
                                     bg="#0088ff", fg="#ffffff",
                                     activebackground="#0066cc",
                                     command=self._launch_collect_data,
                                     padx=20, pady=15)
        self.collect_btn.pack(fill=tk.X, pady=(0, 15))
        
        self.train_btn = tk.Button(right_inner,
                                   text="🎓 TRAIN MODEL",
                                   font=("Helvetica", 13, "bold"),
                                   bg="#ffaa00", fg="#000000",
                                   activebackground="#ff8800",
                                   command=self._launch_train_model,
                                   padx=20, pady=15)
        self.train_btn.pack(fill=tk.X)
    
    def _update_status(self):
        self.model_exists = os.path.exists(MODEL_FILE)
        self.dataset_exists = os.path.exists(DATASET_FILE)
        
        if self.dataset_exists:
            dataset_size = os.path.getsize(DATASET_FILE) / 1024
            self.dataset_status_label.config(
                text=f"● Dataset: Ready ({dataset_size:.1f} KB)",
                fg="#00ff00"
            )
            self.train_btn.config(state=tk.NORMAL, bg="#ffaa00", fg="#000000")
        else:
            self.dataset_status_label.config(
                text="● Dataset: Not Found",
                fg="#ff6600"
            )
            self.train_btn.config(state=tk.DISABLED, bg="#666666", fg="#aaaaaa")
        
        if self.model_exists:
            model_size = os.path.getsize(MODEL_FILE) / 1024
            self.model_status_label.config(
                text=f"● Model: Ready ({model_size:.1f} KB)",
                fg="#00ff00"
            )
            self.run_app_btn.config(state=tk.NORMAL, bg="#00ff00", fg="#000000")
        else:
            self.model_status_label.config(
                text="● Model: Not Found",
                fg="#ff0000"
            )
            self.run_app_btn.config(state=tk.NORMAL, bg="#ff6600", fg="#ffffff")
    
    def _start_auto_refresh(self):
        self._update_status()
        self.root.after(2000, self._start_auto_refresh)
    
    def _launch_desktop_app(self):
        try:
            subprocess.Popen([sys.executable, DESKTOP_APP_FILE])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch desktop app:\n{str(e)}", parent=self.root)
    
    def _launch_collect_data(self):
        try:
            subprocess.Popen([sys.executable, COLLECT_DATA_FILE])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch collect_data:\n{str(e)}", parent=self.root)
    
    def _launch_train_model(self):
        try:
            subprocess.Popen([sys.executable, TRAIN_MODEL_FILE])
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch train_model:\n{str(e)}", parent=self.root)


if __name__ == '__main__':
    root = tk.Tk()
    app = LauncherDevApp(root)
    root.mainloop()
