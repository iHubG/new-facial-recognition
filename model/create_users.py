import sqlite3
from datetime import datetime
import os

def create_table():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    db_path = os.path.join(current_dir, '..', 'face-recognition.db') 
    
    conn = sqlite3.connect(db_path) 
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            grade_level TEXT,
            section TEXT,
            user_type TEXT NOT NULL,
            total_attendance INTEGER DEFAULT 0,
            weekly_attendance INTEGER,
            week INTEGER,
            date_created TEXT 
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table() 