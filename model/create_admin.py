import sqlite3
import bcrypt
from pathlib import Path

def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_admin():
    db_path = 'face.db'
    
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            ''')

            cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
            if cursor.fetchone()[0] > 0:
                print("Admin user already exists.")
                return

            username = 'admin'
            password = 'Admin12345'
            hashed_password = hash_password(password)

            cursor.execute('''
            INSERT INTO users (username, password_hash)
            VALUES (?, ?)
            ''', (username, hashed_password))

            conn.commit()
            print("Admin user created successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    create_admin()
