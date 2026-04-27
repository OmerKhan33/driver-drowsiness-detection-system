"""
Database module for Driver Drowsiness Detection System.

Provides SQLite-based storage for user authentication,
session tracking, and drowsiness event logging.
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path


# ─── Database Path ────────────────────────────────────────────────────────────

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "driver_drowsiness.db"


# ─── Connection Helper ───────────────────────────────────────────────────────


def get_connection() -> sqlite3.Connection:
    """Get a SQLite database connection.

    Creates the database directory and file if they don't exist.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ─── Schema Initialization ───────────────────────────────────────────────────


def init_db() -> None:
    """Initialize database tables and seed the admin user.

    Creates users, sessions, and events tables if they don't exist.
    Seeds a default admin account (admin / admin123).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            role        TEXT    NOT NULL DEFAULT 'driver',
            full_name   TEXT    NOT NULL DEFAULT '',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            driver_id       INTEGER NOT NULL,
            start_time      TEXT    NOT NULL,
            end_time        TEXT,
            total_frames    INTEGER DEFAULT 0,
            drowsy_frames   INTEGER DEFAULT 0,
            total_alerts    INTEGER DEFAULT 0,
            total_yawns     INTEGER DEFAULT 0,
            avg_ear         REAL    DEFAULT 0.0,
            avg_mar         REAL    DEFAULT 0.0,
            status          TEXT    DEFAULT 'active',
            FOREIGN KEY (driver_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            driver_id   INTEGER NOT NULL,
            event_type  TEXT    NOT NULL,
            ear_value   REAL    DEFAULT 0.0,
            mar_value   REAL    DEFAULT 0.0,
            score       REAL    DEFAULT 0.0,
            consec      INTEGER DEFAULT 0,
            timestamp   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (driver_id)  REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_driver
            ON events(driver_id);
        CREATE INDEX IF NOT EXISTS idx_events_session
            ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_driver
            ON sessions(driver_id);
    """)

    # Seed admin user if not exists
    existing = cursor.execute(
        "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
    ).fetchone()
    if not existing:
        admin_pw = hash_password("admin123")
        cursor.execute(
            "INSERT INTO users (username, password, role, full_name) "
            "VALUES (?, ?, 'admin', 'System Administrator')",
            ("admin", admin_pw),
        )

    conn.commit()
    conn.close()


# ─── Password Hashing ────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with a salt.

    Args:
        password: Plain text password.

    Returns:
        Salted hash string in the format 'salt$hash'.
    """
    salt = os.urandom(16).hex()
    pw_hash = hashlib.sha256(
        (salt + password).encode("utf-8")
    ).hexdigest()
    return f"{salt}${pw_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: Plain text password to verify.
        stored_hash: Stored hash in 'salt$hash' format.

    Returns:
        True if the password matches, False otherwise.
    """
    if "$" not in stored_hash:
        return False
    salt, expected_hash = stored_hash.split("$", 1)
    actual_hash = hashlib.sha256(
        (salt + password).encode("utf-8")
    ).hexdigest()
    return actual_hash == expected_hash


# ─── User Operations ─────────────────────────────────────────────────────────


def create_user(username: str, password: str, full_name: str) -> dict:
    """Create a new driver user.

    Args:
        username: Unique username.
        password: Plain text password (will be hashed).
        full_name: Driver's full name.

    Returns:
        Dict with 'success' bool and 'message' or 'user_id'.
    """
    conn = get_connection()
    try:
        pw_hash = hash_password(password)
        cursor = conn.execute(
            "INSERT INTO users (username, password, role, full_name) "
            "VALUES (?, ?, 'driver', ?)",
            (username, pw_hash, full_name),
        )
        conn.commit()
        return {"success": True, "user_id": cursor.lastrowid}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Username already exists"}
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict:
    """Authenticate a user by username and password.

    Args:
        username: Username to look up.
        password: Plain text password to verify.

    Returns:
        Dict with 'success' bool and user info on success.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return {"success": False, "message": "User not found"}

    if not verify_password(password, row["password"]):
        return {"success": False, "message": "Incorrect password"}

    return {
        "success": True,
        "user_id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "full_name": row["full_name"],
    }


def get_all_drivers() -> list[dict]:
    """Get all driver users.

    Returns:
        List of dicts with driver info.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, full_name, created_at "
        "FROM users WHERE role = 'driver' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Session Operations ──────────────────────────────────────────────────────


def start_session(driver_id: int) -> int:
    """Start a new monitoring session.

    Args:
        driver_id: ID of the driver.

    Returns:
        The new session ID.
    """
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor = conn.execute(
        "INSERT INTO sessions (driver_id, start_time, status) "
        "VALUES (?, ?, 'active')",
        (driver_id, now),
    )
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id


def end_session(
    session_id: int,
    total_frames: int,
    drowsy_frames: int,
    total_alerts: int,
    total_yawns: int,
    avg_ear: float,
    avg_mar: float,
) -> None:
    """End a monitoring session and save summary stats.

    Args:
        session_id: Session to close.
        total_frames: Total frames processed.
        drowsy_frames: Frames flagged as drowsy.
        total_alerts: Number of alerts triggered.
        total_yawns: Number of yawns detected.
        avg_ear: Average EAR during the session.
        avg_mar: Average MAR during the session.
    """
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE sessions SET end_time=?, total_frames=?, drowsy_frames=?, "
        "total_alerts=?, total_yawns=?, avg_ear=?, avg_mar=?, status='completed' "
        "WHERE id=?",
        (now, total_frames, drowsy_frames, total_alerts,
         total_yawns, avg_ear, avg_mar, session_id),
    )
    conn.commit()
    conn.close()


# ─── Event Operations ────────────────────────────────────────────────────────


def log_event(
    session_id: int,
    driver_id: int,
    event_type: str,
    ear: float = 0.0,
    mar: float = 0.0,
    score: float = 0.0,
    consec: int = 0,
) -> None:
    """Log a drowsiness event.

    Args:
        session_id: Current session ID.
        driver_id: Driver ID.
        event_type: One of 'drowsy', 'yawn', 'alert'.
        ear: EAR at the time of the event.
        mar: MAR at the time of the event.
        score: Drowsiness score at the time.
        consec: Consecutive drowsy frames count.
    """
    conn = get_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO events "
        "(session_id, driver_id, event_type, ear_value, "
        "mar_value, score, consec, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, driver_id, event_type, ear, mar, score, consec, now),
    )
    conn.commit()
    conn.close()


def get_driver_events(
    driver_id: int = None,
    limit: int = 200,
) -> list[dict]:
    """Get events, optionally filtered by driver.

    Args:
        driver_id: If provided, filter events to this driver.
        limit: Max events to return.

    Returns:
        List of event dicts with driver name joined.
    """
    conn = get_connection()
    if driver_id:
        rows = conn.execute(
            "SELECT e.*, u.full_name, u.username "
            "FROM events e JOIN users u ON e.driver_id = u.id "
            "WHERE e.driver_id = ? "
            "ORDER BY e.timestamp DESC LIMIT ?",
            (driver_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT e.*, u.full_name, u.username "
            "FROM events e JOIN users u ON e.driver_id = u.id "
            "ORDER BY e.timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_sessions(driver_id: int = None) -> list[dict]:
    """Get all sessions, optionally filtered by driver.

    Args:
        driver_id: If provided, filter to this driver.

    Returns:
        List of session dicts with driver name joined.
    """
    conn = get_connection()
    if driver_id:
        rows = conn.execute(
            "SELECT s.*, u.full_name, u.username "
            "FROM sessions s JOIN users u ON s.driver_id = u.id "
            "WHERE s.driver_id = ? "
            "ORDER BY s.start_time DESC",
            (driver_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT s.*, u.full_name, u.username "
            "FROM sessions s JOIN users u ON s.driver_id = u.id "
            "ORDER BY s.start_time DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_stats() -> dict:
    """Get aggregate statistics for the admin dashboard.

    Returns:
        Dict with total drivers, sessions, alerts, and yawns.
    """
    conn = get_connection()
    drivers = conn.execute(
        "SELECT COUNT(*) FROM users WHERE role='driver'"
    ).fetchone()[0]
    sessions = conn.execute(
        "SELECT COUNT(*) FROM sessions"
    ).fetchone()[0]
    alerts = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='drowsy'"
    ).fetchone()[0]
    yawns = conn.execute(
        "SELECT COUNT(*) FROM events WHERE event_type='yawn'"
    ).fetchone()[0]
    conn.close()
    return {
        "total_drivers": drivers,
        "total_sessions": sessions,
        "total_alerts": alerts,
        "total_yawns": yawns,
    }


# ─── Init on import ──────────────────────────────────────────────────────────

init_db()
