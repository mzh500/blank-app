# database.py
import hashlib
import sqlite3
import os

DB_NAME = 'detection_system.db'


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    if not os.path.exists(DB_NAME):
        conn = get_db_connection()
        # 创建用户表
        conn.execute('''
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            )
        ''')
        # 创建检测记录表
        conn.execute('''
            CREATE TABLE detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                time TEXT NOT NULL,
                file_path TEXT NOT NULL,
                detect_label TEXT,
                confidence REAL,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        ''')
        # 添加默认管理员
        conn.execute('''
            INSERT INTO users (username, password_hash)
            VALUES (?, ?)
        ''', ('admin', hash_password('123')))
        conn.commit()
        conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()
