import sqlite3
import json
import datetime
from typing import Dict, List
import os

class ExperimentLogger:
    """
    Log all mix experiments to database for research analysis
    """
    
    def __init__(self, db_path='experiments.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                num_tracks INTEGER,
                total_duration REAL,
                stem_separation BOOLEAN,
                tempo_matching BOOLEAN,
                harmonic_matching BOOLEAN,
                settings TEXT
            )
        ''')
        
        # Transitions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                track_a TEXT,
                track_b TEXT,
                pattern_match_score REAL,
                transition_type TEXT,
                bpm_a REAL,
                bpm_b REAL,
                energy_a REAL,
                energy_b REAL,
                smoothness_score REAL,
                beat_alignment_error_ms REAL,
                loudness_continuity REAL,
                transition_start_sec REAL,
                transition_end_sec REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        ''')
        
        # Feature vectors table (for ML training)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transition_id INTEGER,
                feature_json TEXT,
                quality_label INTEGER,
                FOREIGN KEY (transition_id) REFERENCES transitions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def log_experiment(self, settings: Dict) -> int:
        """Log a new experiment and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments (
                timestamp, session_id, num_tracks, total_duration,
                stem_separation, tempo_matching, harmonic_matching, settings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            settings.get('session_id'),
            settings.get('num_tracks'),
            settings.get('total_duration'),
            settings.get('stem_separation', False),
            settings.get('tempo_matching', False),
            settings.get('harmonic_matching', False),
            json.dumps(settings)
        ))
        
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"📝 Logged experiment #{experiment_id}")
        return experiment_id
    
    def log_transition(self, experiment_id: int, transition_data: Dict):
        """Log a single transition"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transitions (
                experiment_id, track_a, track_b, pattern_match_score,
                transition_type, bpm_a, bpm_b, energy_a, energy_b,
                smoothness_score, beat_alignment_error_ms, loudness_continuity,
                transition_start_sec, transition_end_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id,
            transition_data.get('track_a'),
            transition_data.get('track_b'),
            transition_data.get('pattern_match_score'),
            transition_data.get('transition_type'),
            transition_data.get('bpm_a'),
            transition_data.get('bpm_b'),
            transition_data.get('energy_a'),
            transition_data.get('energy_b'),
            transition_data.get('smoothness_score'),
            transition_data.get('beat_alignment_error_ms'),
            transition_data.get('loudness_continuity'),
            transition_data.get('transition_start_sec'),
            transition_data.get('transition_end_sec')
        ))
        
        transition_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return transition_id
    
    def export_for_analysis(self, output_csv='experiments_export.csv'):
        """Export all data to CSV for statistical analysis"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                e.id, e.timestamp, e.num_tracks, e.stem_separation,
                t.track_a, t.track_b, t.pattern_match_score,
                t.bpm_a, t.bpm_b, t.smoothness_score,
                t.beat_alignment_error_ms, t.loudness_continuity
            FROM experiments e
            JOIN transitions t ON e.id = t.experiment_id
        '''
        
        import pandas as pd
        df = pd.read_sql_query(query, conn)
        df.to_csv(output_csv, index=False)
        conn.close()
        
        print(f"📊 Exported {len(df)} transitions to {output_csv}")
        return output_csv
