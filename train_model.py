""" 
MODEL TRAINER - Sign Language Detection System
Complete rewrite with proper path handling and error visibility
"""

import tkinter as tk
from tkinter import ttk
import threading
import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# ============================================================
# PATH CONFIGURATION - Use sys.executable location ONLY
# ============================================================
from path_config import (
    PROJECT_ROOT, BUILT_IN_DIR, DATA_ROOT, 
    DATASET_FILE, MODEL_FILE
)
print(f"[PATH CONFIG] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[PATH CONFIG] DATA_ROOT: {DATA_ROOT}")

CREATE_DATASET_SCRIPT = os.path.join(BUILT_IN_DIR, 'create_dataset.py')

# Add built-in to Python path
sys.path.insert(0, BUILT_IN_DIR)


class ModelTrainerGUI:
    """
    Professional Model Training Interface
    - Dark theme matching desktop_app.py
    - Real-time log output with error visibility
    - Proper path handling for built-in/ and data/
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("MODEL TRAINER - Sign Language Detection")
        self.root.geometry("950x750")
        self.root.configure(bg="#1a1a1a")
        
        # Apply dark theme
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except tk.TclError:
            pass
        
        # GUI state
        self.status_msg = tk.StringVar(value="Ready. Click Step 1 to create dataset.")
        self.log_text = None
        self.create_btn = None
        self.train_btn = None
        
        self._build_ui()
        self._verify_paths()
    
    def _build_ui(self):
        """Build the complete user interface"""
        main_container = tk.Frame(self.root, bg="#1a1a1a", padx=20, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # ========== HEADER ==========
        header = tk.Label(main_container, 
                         text="🎓 MODEL TRAINING SYSTEM",
                         font=("Helvetica", 16, "bold"),
                         fg="#00ff88", bg="#1a1a1a")
        header.pack(pady=(0, 20))
        
        # ========== WORKFLOW SECTION ==========
        workflow_frame = tk.Frame(main_container, bg="#2b2b2b", relief=tk.RAISED, bd=2)
        workflow_frame.pack(fill=tk.X, pady=(0, 15))
        
        workflow_inner = tk.Frame(workflow_frame, bg="#2b2b2b", padx=15, pady=15)
        workflow_inner.pack(fill=tk.BOTH)
        
        tk.Label(workflow_inner, text="MODEL TRAINING",
                font=("Helvetica", 11, "bold"),
                fg="#00ff88", bg="#2b2b2b").pack(anchor=tk.W)
        
        tk.Frame(workflow_inner, height=2, bg="#00ff88").pack(fill=tk.X, pady=(5, 15))
        
        # Info label
        tk.Label(workflow_inner,
                text="Dataset is created automatically after data collection.\n"
                     "Click the button below to train the recognition model.",
                font=("Helvetica", 9),
                fg="#aaaaaa", bg="#2b2b2b",
                justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 15))
        
        # Train Button (always enabled if dataset exists)
        self.train_btn = tk.Button(workflow_inner,
                                   text="TRAIN MODEL",
                                   font=("Helvetica", 12, "bold"),
                                   bg="#00ff88", fg="#000000",
                                   activebackground="#00dd66",
                                   command=self._run_train_model,
                                   padx=20, pady=20)
        self.train_btn.pack(fill=tk.X)
        
        # ========== LOG SECTION ==========
        log_frame = tk.Frame(main_container, bg="#2b2b2b", relief=tk.RAISED, bd=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        log_inner = tk.Frame(log_frame, bg="#2b2b2b", padx=15, pady=15)
        log_inner.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(log_inner, text="SYSTEM LOG",
                font=("Helvetica", 11, "bold"),
                fg="#ff9500", bg="#2b2b2b").pack(anchor=tk.W)
        
        tk.Frame(log_inner, height=2, bg="#ff9500").pack(fill=tk.X, pady=(5, 10))
        
        # Status label
        tk.Label(log_inner, textvariable=self.status_msg,
                font=("Helvetica", 10, "bold"),
                fg="#ffff00", bg="#2b2b2b",
                wraplength=880, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 10))
        
        # Log text widget
        log_container = tk.Frame(log_inner, bg="#0a0a0a", relief=tk.SUNKEN, bd=2)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_container, 
                               height=22, wrap="word",
                               bg="#0a0a0a", fg="#00ff00",
                               font=("Consolas", 9),
                               state=tk.DISABLED,
                               yscrollcommand=scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar.config(command=self.log_text.yview)
        
        # Configure tags
        self.log_text.tag_config("INFO", foreground="#00ffff")
        self.log_text.tag_config("SUCCESS", foreground="#00ff00", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("WARN", foreground="#ffaa00")
        self.log_text.tag_config("ERROR", foreground="#ff0000", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("PROCESS", foreground="#00ff88")
        self.log_text.tag_config("DETAIL", foreground="#888888")
        
        # ========== EXIT BUTTON ==========
        tk.Button(main_container, text="Exit",
                 font=("Helvetica", 10, "bold"),
                 bg="#555555", fg="white",
                 command=self.root.destroy,
                 padx=50, pady=10).pack()
    
    def _verify_paths(self):
        """Verify all required paths and log status"""
        self._log("=" * 60, "INFO")
        self._log("SYSTEM INITIALIZATION", "PROCESS")
        self._log("=" * 60, "INFO")
        self._log(f"Project Root: {PROJECT_ROOT}", "INFO")
        self._log(f"Built-in Dir: {BUILT_IN_DIR}", "INFO")
        self._log(f"Data Dir: {DATA_ROOT}", "INFO")
        self._log(f"Create Dataset Script: {CREATE_DATASET_SCRIPT}", "INFO")
        self._log("=" * 60, "INFO")
        
        # Check paths
        if not os.path.exists(BUILT_IN_DIR):
            self._log(f"WARNING: built-in directory not found!", "ERROR")
        else:
            self._log("✓ built-in directory found", "SUCCESS")
        
        if not os.path.exists(DATA_ROOT):
            self._log(f"WARNING: Data directory not found at {DATA_ROOT}", "WARN")
            self._log("Please run collect_data.py first to collect training data", "WARN")
        else:
            self._log("✓ Data directory found", "SUCCESS")
        
        if not os.path.exists(CREATE_DATASET_SCRIPT):
            self._log(f"ERROR: create_dataset.py not found!", "ERROR")
        else:
            self._log("✓ create_dataset.py found", "SUCCESS")
        
        self._log("=" * 60, "INFO")
    
    def _log(self, message, level="INFO"):
        """Thread-safe logging to GUI"""
        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{level}] {message}\n", level)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.root.after(0, update)

    
    def _run_train_model(self):
        """Run model training in background thread"""
        self.train_btn.config(state=tk.DISABLED, bg="#444444", fg="#888888")
        threading.Thread(target=self._train_model_worker, daemon=True).start()
    
    def _train_model_worker(self):
        """Worker thread for model training"""
        self._log("=" * 60, "PROCESS")
        self._log("TRAINING MACHINE LEARNING MODEL", "PROCESS")
        self._log("=" * 60, "PROCESS")
        self.status_msg.set("Training model... This may take a few minutes.")
        
        try:
            # Verify dataset exists
            if not os.path.exists(DATASET_FILE):
                self._log(f"ERROR: Dataset not found: {DATASET_FILE}", "ERROR")
                self._log("Please run Step 1 first", "ERROR")
                self.status_msg.set("Failed: Dataset not found")
                return
            
            # Load dataset
            self._log("Loading dataset...", "INFO")
            with open(DATASET_FILE, 'rb') as f:
                dataset = pickle.load(f)
            
            data = np.asarray(dataset['data'])
            labels = np.asarray(dataset['labels'])
            
            self._log(f"✓ Loaded {len(data)} samples", "SUCCESS")
            self._log(f"✓ Found {len(set(labels))} unique classes", "SUCCESS")
            
            # Show class distribution
            class_counts = Counter(labels)
            self._log("\nClass distribution:", "INFO")
            for label, count in sorted(class_counts.items()):
                self._log(f"  {label}: {count} samples", "DETAIL")
            
            # Split data
            self._log("\nSplitting data (80% train, 20% test)...", "INFO")
            
            min_samples = 2
            use_stratify = all(count >= min_samples for count in class_counts.values())
            
            if use_stratify:
                self._log("Using stratified split", "INFO")
                x_train, x_test, y_train, y_test = train_test_split(
                    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
                )
            else:
                self._log("Warning: Some classes have <2 samples, using random split", "WARN")
                x_train, x_test, y_train, y_test = train_test_split(
                    data, labels, test_size=0.2, shuffle=True, random_state=42
                )
            
            self._log(f"✓ Training set: {len(x_train)} samples", "SUCCESS")
            self._log(f"✓ Testing set: {len(x_test)} samples", "SUCCESS")
            
            # Train model
            self._log("\nTraining Random Forest Classifier...", "PROCESS")
            self._log("Parameters: 200 trees, max depth 25", "INFO")
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            model.fit(x_train, y_train)
            self._log("✓ Training complete", "SUCCESS")
            
            # Evaluate
            self._log("\nEvaluating model...", "INFO")
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self._log(f"✓ Test Accuracy: {accuracy * 100:.2f}%", "SUCCESS")
            
            # Classify static vs dynamic
            static_classes = sorted([l for l in set(labels) if len(l) == 1 or l.isdigit()])
            dynamic_classes = sorted([l for l in set(labels) if len(l) > 1 and not l.isdigit()])
            
            self._log(f"\nStatic classes ({len(static_classes)}): {', '.join(static_classes)}", "INFO")
            if dynamic_classes:
                self._log(f"Dynamic classes ({len(dynamic_classes)}): {', '.join(dynamic_classes)}", "INFO")
            
            # Save model
            self._log("\nSaving model...", "INFO")
            os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
            
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'labels_dict': {i: label for i, label in enumerate(sorted(set(labels)))},
                    'static_classes': static_classes,
                    'dynamic_classes': dynamic_classes
                }, f)
            
            model_size = os.path.getsize(MODEL_FILE) / 1024
            
            self._log("=" * 60, "SUCCESS")
            self._log("✓ MODEL TRAINING SUCCESSFUL", "SUCCESS")
            self._log(f"✓ Model saved to: {MODEL_FILE}", "SUCCESS")
            self._log(f"✓ File size: {model_size:.2f} KB", "SUCCESS")
            self._log(f"✓ Final Accuracy: {accuracy * 100:.2f}%", "SUCCESS")
            self._log("=" * 60, "SUCCESS")
            
            self.status_msg.set(f"Training Complete! Accuracy: {accuracy * 100:.2f}%")
            
        except Exception as e:
            import traceback
            self._log("=" * 60, "ERROR")
            self._log("TRAINING FAILED", "ERROR")
            self._log(f"Error: {str(e)}", "ERROR")
            self._log("\nFull traceback:", "ERROR")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self._log(line, "ERROR")
            self._log("=" * 60, "ERROR")
            self.status_msg.set("Training failed - see log")
        
        finally:
            # Re-enable buttons
            self.root.after(0, lambda: self.create_btn.config(
                state=tk.NORMAL, bg="#0088ff", fg="white"
            ))
            self.root.after(0, lambda: self.train_btn.config(
                state=tk.DISABLED, bg="#666666", fg="#aaaaaa"
            )) 

if __name__ == '__main__':
    root = tk.Tk()
    app = ModelTrainerGUI(root)
    root.mainloop()