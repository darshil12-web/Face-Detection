import sqlite3

def init_db():
    conn = sqlite3.connect("face_finder.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            threshold REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            file_data BLOB,
            uploaded_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            target_face_data BLOB,
            target_name TEXT,
            matched_photos TEXT,
            detected_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)

    conn.commit()
    return conn
