import sqlite3
from datetime import datetime
import os

def create_table():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    db_path = os.path.join(current_dir, '..', 'face-recognition.db') 
    
    conn = sqlite3.connect(db_path) 
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,  
            section TEXT, 
            grade_level TEXT,
            time_in TEXT,
            time_out TEXT
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table() 