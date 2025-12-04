"""
Session Logger - Records all detection events for analysis
Daily logging with weekly report generation
"""

import os
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import glob

class SessionLogger:
    def __init__(self, log_dir="logs", enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Use only date for session ID (YYYYMMDD format)
        today = datetime.now().strftime("%Y%m%d")
        self.session_id = f"session_{today}"
        
        # Create or reuse daily session folder
        self.session_path = self.log_dir / self.session_id
        self.session_path.mkdir(exist_ok=True)
        
        # CSV file for structured data
        self.csv_path = self.session_path / "detections.csv"
        
        # Check if file exists (append mode) or create new
        file_exists = self.csv_path.exists()
        self.csv_file = open(self.csv_path, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header only if new file
        if not file_exists:
            self.csv_writer.writerow([
                'Timestamp', 'Frame_Number', 'Detected_Letter', 'Confidence', 
                'Current_Word', 'Sentence', 'Hand_Position', 'Processing_Time_Ms'
            ])
        
        # Load or create metadata
        self.metadata_path = self.session_path / "summary.json"
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"[INFO] Resuming daily log: {self.session_id}")
        else:
            self.metadata = {
                'session_id': self.session_id,
                'date': today,
                'start_time': datetime.now().isoformat(),
                'statistics': {
                    'total_frames': 0,
                    'hands_detected': 0,
                    'letters_recognized': 0,
                    'words_formed': 0
                }
            }
            print(f"[INFO] Daily logging started: {self.session_id}")
        
        # Check for weekly report generation
        self._check_weekly_report()
    
    def log_detection(self, frame_num, letter, confidence, word, sentence, position, proc_time):
        """Log a single detection event"""
        if not self.enabled:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Write to CSV
        self.csv_writer.writerow([
            timestamp, frame_num, letter, f"{confidence:.2f}", 
            word, sentence, position, f"{proc_time:.2f}"
        ])
        self.csv_file.flush()
        
        # Update statistics
        self.metadata['statistics']['total_frames'] = frame_num
        if letter and letter != "--":
            self.metadata['statistics']['letters_recognized'] += 1
    
    def log_word_formed(self, word):
        """Log when a word is completed"""
        if not self.enabled or not word:
            return
        self.metadata['statistics']['words_formed'] += 1
    
    def log_hand_detected(self):
        """Increment hand detection counter"""
        if not self.enabled:
            return
        self.metadata['statistics']['hands_detected'] += 1
    
    def save_summary(self):
        """Save daily summary as JSON"""
        if not self.enabled:
            return
            
        self.metadata['last_update'] = datetime.now().isoformat()
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Daily session saved: {self.session_path}")
        print(f"  - Detections: {self.metadata['statistics']['letters_recognized']}")
        print(f"  - Words: {self.metadata['statistics']['words_formed']}")
    
    def _check_weekly_report(self):
        """Check if 7 days of logs exist and generate weekly report"""
        # Find all session folders
        session_folders = sorted(glob.glob(str(self.log_dir / "session_*")))
        
        # Filter to only daily sessions (8-digit dates: session_YYYYMMDD)
        daily_sessions = []
        for folder in session_folders:
            folder_name = Path(folder).name
            if folder_name.startswith("session_") and len(folder_name) == 16:  # session_YYYYMMDD
                try:
                    date_str = folder_name.split("_")[1]
                    datetime.strptime(date_str, "%Y%m%d")  # Validate date format
                    daily_sessions.append(folder)
                except:
                    continue
        
        # Check if we have 7 or more days
        if len(daily_sessions) >= 7:
            # Get last 7 sessions
            recent_sessions = daily_sessions[-7:]
            
            # Check if weekly report already exists for this period
            first_date = Path(recent_sessions[0]).name.split("_")[1]
            last_date = Path(recent_sessions[-1]).name.split("_")[1]
            report_name = f"weekly_report_{first_date}_to_{last_date}.json"
            report_path = self.log_dir / report_name
            
            if not report_path.exists():
                self._generate_weekly_report(recent_sessions, report_path)
    
    def _generate_weekly_report(self, session_folders, report_path):
        """Generate weekly average report from 7 days of data"""
        print(f"[INFO] Generating weekly report from {len(session_folders)} days...")
        
        weekly_data = {
            'report_type': 'weekly',
            'period': {
                'start_date': Path(session_folders[0]).name.split("_")[1],
                'end_date': Path(session_folders[-1]).name.split("_")[1]
            },
            'daily_sessions': [],
            'averages': {
                'avg_letters_per_day': 0,
                'avg_words_per_day': 0,
                'avg_hands_detected_per_day': 0,
                'total_letters': 0,
                'total_words': 0,
                'total_hands_detected': 0
            },
            'generated_at': datetime.now().isoformat()
        }
        
        total_letters = 0
        total_words = 0
        total_hands = 0
        valid_days = 0
        
        # Aggregate data from all sessions
        for folder in session_folders:
            summary_path = Path(folder) / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    stats = session_data.get('statistics', {})
                    letters = stats.get('letters_recognized', 0)
                    words = stats.get('words_formed', 0)
                    hands = stats.get('hands_detected', 0)
                    
                    total_letters += letters
                    total_words += words
                    total_hands += hands
                    valid_days += 1
                    
                    weekly_data['daily_sessions'].append({
                        'date': session_data.get('date', Path(folder).name.split("_")[1]),
                        'letters': letters,
                        'words': words,
                        'hands_detected': hands
                    })
                except:
                    continue
        
        # Calculate averages
        if valid_days > 0:
            weekly_data['averages']['avg_letters_per_day'] = round(total_letters / valid_days, 2)
            weekly_data['averages']['avg_words_per_day'] = round(total_words / valid_days, 2)
            weekly_data['averages']['avg_hands_detected_per_day'] = round(total_hands / valid_days, 2)
            weekly_data['averages']['total_letters'] = total_letters
            weekly_data['averages']['total_words'] = total_words
            weekly_data['averages']['total_hands_detected'] = total_hands
            weekly_data['days_analyzed'] = valid_days
        
        # Save weekly report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(weekly_data, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Weekly report generated: {report_path}")
        print(f"  - Period: {weekly_data['period']['start_date']} to {weekly_data['period']['end_date']}")
        print(f"  - Avg Letters/Day: {weekly_data['averages']['avg_letters_per_day']}")
        print(f"  - Avg Words/Day: {weekly_data['averages']['avg_words_per_day']}")
        print(f"  - Total Letters: {weekly_data['averages']['total_letters']}")
        print(f"  - Total Words: {weekly_data['averages']['total_words']}")
    
    def close(self):
        """Close log files"""
        if not self.enabled:
            return
        self.csv_file.close()
        self.save_summary()
