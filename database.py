import sqlite3

def init_db():
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            encoding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database ready.")

def save_user(name, encoding):
    import numpy as np
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, encoding) VALUES (?, ?)",
            (name, encoding.tobytes())
        )
        conn.commit()
        print(f"{name} registered successfully.")
    except sqlite3.IntegrityError:
        print(f"{name} already exists.")
    finally:
        conn.close()

def get_all_users():
    import numpy as np
    conn = sqlite3.connect("database/users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM users")
    rows = cursor.fetchall()
    conn.close()
    
    users = []
    for name, enc_bytes in rows:
        encoding = np.frombuffer(enc_bytes, dtype=np.float32)
        users.append((name, encoding))
    return users

if __name__ == "__main__":
    init_db()