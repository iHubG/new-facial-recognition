import sqlite3
from datetime import datetime

def create_table():
    conn = sqlite3.connect('face.db') 
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activitylogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            activity TEXT NOT NULL,
            date_time TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table()  




