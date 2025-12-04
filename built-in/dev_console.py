import tkinter as tk
from tkinter import scrolledtext
import time
from datetime import datetime

class DeveloperConsole:
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.console_window = tk.Toplevel()
        self.console_window.title("Developer Console")
        self.console_window.geometry("700x500")
        self.console_window.configure(bg="#1a1a1a")
        
        # Keep window on top
        self.console_window.attributes('-topmost', False)
        
        # Create UI
        self.create_widgets()
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        print("[INFO] Developer Console initialized")
    
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.console_window, bg="#1a1a1a", padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = tk.Label(main_frame, text="🛠️ DEVELOPER CONSOLE", 
                         font=("Consolas", 14, "bold"), fg="#00ff00", bg="#1a1a1a")
        header.pack(pady=(0, 10))
        
        # Stats Frame
        stats_frame = tk.Frame(main_frame, bg="#2b2b2b", relief=tk.RIDGE, borderwidth=2)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_inner = tk.Frame(stats_frame, bg="#2b2b2b", padx=10, pady=10)
        stats_inner.pack(fill=tk.BOTH)
        
        # Create stat labels
        self.stat_labels = {}
        
        stats_data = [
            ("uptime", "Uptime:", "0s"),
            ("fps", "Current FPS:", "0"),
            ("frames", "Total Frames:", "0"),
            ("detections", "Detections:", "0"),
            ("landmarks", "Landmarks:", "0"),
            ("handedness", "Hand Type:", "None"),
            ("position", "Position:", "(0, 0)"),
            ("gesture", "Current Gesture:", "NONE"),
            ("confidence", "Confidence:", "0%"),
            ("mqtt_status", "MQTT Status:", "Disconnected"),
        ]
        
        for key, label, default in stats_data:
            row = tk.Frame(stats_inner, bg="#2b2b2b")
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(row, text=label, font=("Consolas", 9, "bold"), 
                    fg="#888888", bg="#2b2b2b", width=18, anchor=tk.W).pack(side=tk.LEFT)
            
            value_label = tk.Label(row, text=default, font=("Consolas", 9), 
                                  fg="#00ff00", bg="#2b2b2b", anchor=tk.W)
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.stat_labels[key] = value_label
        
        # Log Frame
        log_frame = tk.Frame(main_frame, bg="#2b2b2b", relief=tk.RIDGE, borderwidth=2)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        log_header = tk.Label(log_frame, text="📋 Event Log", 
                             font=("Consolas", 10, "bold"), fg="#00d4ff", bg="#2b2b2b")
        log_header.pack(anchor=tk.W, padx=10, pady=(5, 5))
        
        # Scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg="#0a0a0a",
            fg="#00ff00",
            insertbackground="#00ff00",
            height=15,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg="#1a1a1a")
        button_frame.pack(fill=tk.X)
        
        tk.Button(button_frame, text="Clear Log", command=self.clear_log,
                 bg="#ff3366", fg="white", font=("Consolas", 9, "bold"),
                 cursor="hand2", relief=tk.RAISED, borderwidth=2).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(button_frame, text="Export Log", command=self.export_log,
                 bg="#3366ff", fg="white", font=("Consolas", 9, "bold"),
                 cursor="hand2", relief=tk.RAISED, borderwidth=2).pack(side=tk.LEFT)
        
        # Initial log
        self.add_log("Developer console started")
    
    def update_stats(self, **kwargs):
        """Update statistics display"""
        # Update uptime
        uptime = int(time.time() - self.start_time)
        self.stat_labels['uptime'].config(text=f"{uptime}s")
        
        # Update provided stats
        for key, value in kwargs.items():
            if key in self.stat_labels:
                self.stat_labels[key].config(text=str(value))
        
        # Update counters
        if 'frame_count' in kwargs:
            self.frame_count = kwargs['frame_count']
            self.stat_labels['frames'].config(text=str(self.frame_count))
        
        if 'detection_count' in kwargs:
            self.detection_count = kwargs['detection_count']
            self.stat_labels['detections'].config(text=str(self.detection_count))
    
    def add_log(self, message, level="INFO"):
        """Add a log entry"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Color based on level
        colors = {
            "INFO": "#00ff00",
            "WARN": "#ffaa00",
            "ERROR": "#ff0000",
            "DEBUG": "#00ffff"
        }
        color = colors.get(level, "#ffffff")
        
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Add to text widget
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)  # Auto-scroll
        
        # Limit log size
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete('1.0', '500.0')
    
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete('1.0', tk.END)
        self.add_log("Log cleared")
    
    def export_log(self):
        """Export log to file"""
        try:
            filename = f"dev_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(self.log_text.get('1.0', tk.END))
            self.add_log(f"Log exported to {filename}")
        except Exception as e:
            self.add_log(f"Export failed: {e}", "ERROR")
    
    def close(self):
        """Close the console window"""
        if self.console_window:
            self.console_window.destroy()
