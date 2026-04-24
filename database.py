import sqlite3
import os
from datetime import datetime

DB_FILE = "plates.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Create the plates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_plate(plate_text, image_path):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now()
    cursor.execute('''
        INSERT INTO plates (plate_text, timestamp, image_path)
        VALUES (?, ?, ?)
    ''', (plate_text, timestamp, image_path))
    conn.commit()
    conn.close()
    return cursor.lastrowid
