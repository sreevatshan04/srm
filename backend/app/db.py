"""
MessMate Database Access Layer — PostgreSQL / Neon
Uses Psycopg 3 with connection pooling and SSL mode.
Provides native dictionary and index row access and clean transaction handling.
"""

import os
import sys
import threading
from typing import Optional, Any, List, Dict
from dotenv import load_dotenv

# Ensure environment variables are loaded
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, ".env"))

import psycopg
from psycopg.errors import IntegrityError
from psycopg_pool import ConnectionPool


class DictAndIndexRow(dict):
    """
    Subclasses Python dict to provide native mapping behaviour,
    while also supporting positional index access (e.g. row[0])
    for full compatibility with callers and tests.
    """
    __slots__ = ("_values",)

    def __init__(self, mapping: dict, values: tuple):
        super().__init__(mapping)
        self._values = values

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (int, slice)):
            return self._values[item]
        return super().__getitem__(item)


def dict_and_index_row_factory(cursor: psycopg.Cursor):
    """Row factory for Psycopg 3 returning DictAndIndexRow."""
    if cursor.description is None:
        return lambda values: None
    names = [c.name for c in cursor.description]
    def make_row(values: tuple) -> DictAndIndexRow:
        return DictAndIndexRow(dict(zip(names, values)), values)
    return make_row


class CursorWrapper:
    """
    Thin cursor wrapper around psycopg.Cursor.
    Provides fetchone, fetchall, rowcount, description, and auto-populates
    cur.lastrowid on INSERT queries that include RETURNING id.
    """
    def __init__(self, raw_cursor: psycopg.Cursor):
        self.raw_cursor = raw_cursor
        self.lastrowid: Optional[int] = None
        self._buffered_row: Optional[DictAndIndexRow] = None

    def execute(self, sql: str, params: Optional[Any] = None) -> "CursorWrapper":
        self._buffered_row = None
        self.lastrowid = None

        if params is None:
            self.raw_cursor.execute(sql)
        else:
            self.raw_cursor.execute(sql, params)

        # If this is an INSERT with RETURNING id, capture the generated ID
        if self.raw_cursor.description and sql.strip().upper().startswith("INSERT"):
            col_names = [c.name for c in self.raw_cursor.description]
            if "id" in col_names:
                r = self.raw_cursor.fetchone()
                if r is not None:
                    self._buffered_row = r
                    self.lastrowid = r["id"]

        return self

    def fetchone(self) -> Optional[DictAndIndexRow]:
        if self._buffered_row is not None:
            row = self._buffered_row
            self._buffered_row = None
            return row
        if not self.raw_cursor.description:
            return None
        return self.raw_cursor.fetchone()

    def fetchall(self) -> List[DictAndIndexRow]:
        rows: List[DictAndIndexRow] = []
        if self._buffered_row is not None:
            rows.append(self._buffered_row)
            self._buffered_row = None
        if not self.raw_cursor.description:
            return rows
        rows.extend(self.raw_cursor.fetchall())
        return rows

    @property
    def rowcount(self) -> int:
        return self.raw_cursor.rowcount

    @property
    def description(self):
        return self.raw_cursor.description


class ConnectionWrapper:
    """
    Context manager wrapping a Psycopg 3 connection.
    Supports `with connect() as c:`, automatic commit/rollback,
    and returns connection to the pool on exit.
    """
    def __init__(self, raw_conn: psycopg.Connection, pool: Optional[ConnectionPool] = None):
        self.raw_conn = raw_conn
        self.pool = pool
        self._closed = False

    def __enter__(self) -> "ConnectionWrapper":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                self.rollback()
            else:
                self.commit()
        finally:
            self.close()

    def execute(self, sql: str, params: Optional[Any] = None) -> CursorWrapper:
        cur = self.raw_conn.cursor(row_factory=dict_and_index_row_factory)
        wrapper = CursorWrapper(cur)
        wrapper.execute(sql, params)
        return wrapper

    def commit(self):
        if not self._closed and self.raw_conn and not self.raw_conn.closed:
            self.raw_conn.commit()

    def rollback(self):
        if not self._closed and self.raw_conn and not self.raw_conn.closed:
            self.raw_conn.rollback()

    def close(self):
        if not self._closed:
            self._closed = True
            if self.pool is not None and not self.pool.closed:
                self.pool.putconn(self.raw_conn)
            elif self.raw_conn and not self.raw_conn.closed:
                self.raw_conn.close()


def ensure_sslmode(url: str) -> str:
    """Ensure sslmode=require is present on PostgreSQL / Neon URLs."""
    if not url:
        return url
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


# Thread-safe pool management
_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_database_url() -> str:
    """
    Resolves the active database connection string.
    - If in test mode, strictly requires TEST_DATABASE_URL to prevent touching production Neon.
    - Otherwise, returns DATABASE_URL.
    """
    is_testing = (
        os.getenv("TESTING", "").lower() in ("1", "true", "yes")
        or "pytest" in sys.modules
    )
    if is_testing:
        test_url = os.getenv("TEST_DATABASE_URL", "").strip()
        if not test_url:
            raise RuntimeError(
                "TESTING is active but TEST_DATABASE_URL is not set. "
                "To prevent accidental modifications to the production Neon database, "
                "automated PostgreSQL tests require an isolated TEST_DATABASE_URL."
            )
        # Prevent accidental pointing of TEST_DATABASE_URL to production DATABASE_URL
        prod_url = os.getenv("DATABASE_URL", "").strip()
        if prod_url and test_url == prod_url:
            raise RuntimeError(
                "TEST_DATABASE_URL matches DATABASE_URL! "
                "Automated tests must never target the production Neon database."
            )
        return test_url

    prod_url = os.getenv("DATABASE_URL", "").strip()
    if not prod_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not configured. "
            "Please configure your Neon PostgreSQL connection string in the environment."
        )
    return prod_url


def get_pool() -> ConnectionPool:
    """Initialize or return the singleton ConnectionPool for PostgreSQL."""
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool

    with _pool_lock:
        if _pool is None or _pool.closed:
            raw_url = get_database_url()
            normalized_url = ensure_sslmode(raw_url)
            pool_size = int(os.getenv("DATABASE_POOL_SIZE", "5"))
            _pool = ConnectionPool(
                conninfo=normalized_url,
                min_size=1,
                max_size=pool_size,
                timeout=15.0,
                open=True,
            )
    return _pool


def connect() -> ConnectionWrapper:
    """
    Obtain a database connection context.
    Borrows a connection from the connection pool with SSL mode and row factory enabled.
    """
    pool = get_pool()
    raw_conn = pool.getconn(timeout=15.0)
    return ConnectionWrapper(raw_conn, pool=pool)


# Idempotent PostgreSQL DDL Schema
POSTGRES_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS users(
        id SERIAL PRIMARY KEY,
        firebase_uid TEXT UNIQUE,
        identifier TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('student','manager','reviewer')),
        password_hash TEXT NOT NULL,
        register_no TEXT,
        department TEXT,
        semester TEXT,
        theme TEXT DEFAULT 'light',
        dietary_preference TEXT DEFAULT 'non_veg'
    );""",

    """CREATE TABLE IF NOT EXISTS feedback(
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        meal_id TEXT NOT NULL,
        service_date TEXT NOT NULL,
        rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
        items_json TEXT NOT NULL DEFAULT '{}',
        comment TEXT DEFAULT '',
        category TEXT DEFAULT '',
        dietary TEXT DEFAULT 'non_veg',
        created_at TEXT NOT NULL
    );""",

    """CREATE TABLE IF NOT EXISTS complaints(
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id),
        meal_id TEXT NOT NULL,
        service_date TEXT NOT NULL,
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Submitted',
        processed_for_ai INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );""",

    """CREATE TABLE IF NOT EXISTS issue_groups(
        id SERIAL PRIMARY KEY,
        meal_id TEXT NOT NULL,
        service_date TEXT NOT NULL,
        issue_key TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'Open',
        response TEXT DEFAULT '',
        responded_at TEXT,
        deadline TEXT NOT NULL,
        risk_score INTEGER DEFAULT 0,
        risk_level TEXT DEFAULT 'LOW',
        recurrence_score INTEGER DEFAULT 0,
        delay_score INTEGER DEFAULT 0,
        resolution_credits INTEGER DEFAULT 0,
        distinct_weeks INTEGER DEFAULT 1,
        is_food_safety_hazard INTEGER DEFAULT 0,
        category TEXT DEFAULT 'Food quality',
        first_reported_at TEXT,
        last_reported_at TEXT,
        confidence REAL DEFAULT 0.9,
        UNIQUE(meal_id, service_date, issue_key)
    );""",

    """CREATE TABLE IF NOT EXISTS complaint_group_links(
        complaint_id INTEGER PRIMARY KEY REFERENCES complaints(id) ON DELETE CASCADE,
        group_id INTEGER NOT NULL REFERENCES issue_groups(id) ON DELETE CASCADE
    );""",

    """CREATE TABLE IF NOT EXISTS menu_items(
        id SERIAL PRIMARY KEY,
        menu_date TEXT NOT NULL,
        meal_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        dietary_type TEXT NOT NULL DEFAULT 'Veg'
    );""",

    """CREATE TABLE IF NOT EXISTS issue_week_records(
        id SERIAL PRIMARY KEY,
        group_id INTEGER NOT NULL REFERENCES issue_groups(id) ON DELETE CASCADE,
        week_key TEXT NOT NULL,
        first_seen_at TEXT NOT NULL,
        UNIQUE(group_id, week_key)
    );""",

    """CREATE TABLE IF NOT EXISTS response_cycles(
        id SERIAL PRIMARY KEY,
        group_id INTEGER NOT NULL REFERENCES issue_groups(id) ON DELETE CASCADE,
        cycle_number INTEGER NOT NULL DEFAULT 1,
        started_at TEXT NOT NULL,
        deadline TEXT NOT NULL,
        responded_at TEXT,
        response_text TEXT,
        status TEXT NOT NULL DEFAULT 'Open',
        UNIQUE(group_id, cycle_number)
    );""",

    """CREATE TABLE IF NOT EXISTS resolutions(
        id SERIAL PRIMARY KEY,
        group_id INTEGER NOT NULL REFERENCES issue_groups(id) ON DELETE CASCADE,
        trigger_type TEXT NOT NULL,
        verified_by TEXT NOT NULL,
        credit_applied INTEGER NOT NULL DEFAULT 15,
        notes TEXT DEFAULT '',
        verified_at TEXT NOT NULL
    );""",

    """CREATE TABLE IF NOT EXISTS escalations(
        id SERIAL PRIMARY KEY,
        group_id INTEGER NOT NULL REFERENCES issue_groups(id) ON DELETE CASCADE,
        reason TEXT NOT NULL,
        score_at_trigger INTEGER NOT NULL,
        triggered_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Active',
        closed_at TEXT,
        closed_by TEXT,
        closure_notes TEXT
    );""",

    """CREATE TABLE IF NOT EXISTS risk_history(
        id SERIAL PRIMARY KEY,
        group_id INTEGER NOT NULL REFERENCES issue_groups(id) ON DELETE CASCADE,
        old_score INTEGER NOT NULL,
        new_score INTEGER NOT NULL,
        reason TEXT NOT NULL,
        breakdown_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );""",

    """CREATE TABLE IF NOT EXISTS ai_processing_jobs(
        id SERIAL PRIMARY KEY,
        batch_size INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'completed',
        model TEXT,
        provider TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL
    );""",

    # Indexes
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_firebase_uid ON users(firebase_uid) WHERE firebase_uid IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_complaints_ai ON complaints(processed_for_ai);",
    "CREATE INDEX IF NOT EXISTS idx_complaints_period ON complaints(meal_id, service_date);",
    "CREATE INDEX IF NOT EXISTS idx_feedback_period ON feedback(meal_id, service_date);",
    "CREATE INDEX IF NOT EXISTS idx_issue_groups_period ON issue_groups(meal_id, service_date);",
    "CREATE INDEX IF NOT EXISTS idx_issue_groups_risk ON issue_groups(risk_score, risk_level);",
    "CREATE INDEX IF NOT EXISTS idx_menu_items_date ON menu_items(menu_date, meal_id);",
    "CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status, group_id);",
    "CREATE INDEX IF NOT EXISTS idx_week_records_group ON issue_week_records(group_id, week_key);"
]


def init_db():
    """Execute idempotent PostgreSQL schema creation."""
    with connect() as c:
        for stmt in POSTGRES_SCHEMA_STATEMENTS:
            c.execute(stmt)
        c.commit()
