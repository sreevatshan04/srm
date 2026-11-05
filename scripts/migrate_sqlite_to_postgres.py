#!/usr/bin/env python3
"""
MessMate Data Migration Script: SQLite to PostgreSQL (Neon)
Safely transfers data from backend/messmate.db to Neon PostgreSQL.

Features:
- Opens SQLite strictly READ-ONLY (mode=ro URI) - source can NEVER be modified.
- Respects exact foreign-key dependency order.
- Preserves explicit IDs, relationships, and NULL values.
- Idempotent: uses ON CONFLICT (id) DO NOTHING so it can be re-run safely.
- Synchronizes PostgreSQL primary key sequences (setval) to avoid future ID collisions.
- Supports --dry-run mode to inspect and verify without making changes.
- Validates row counts before and after migration.

USAGE:
  Dry run:
    python scripts/migrate_sqlite_to_postgres.py --dry-run

  Live migration (requires explicit --confirm):
    python scripts/migrate_sqlite_to_postgres.py --confirm

DO NOT RUN THIS SCRIPT UNTIL EXPLICITLY APPROVED BY THE USER.
"""

import os
import sys
import sqlite3
import argparse
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Ensure backend directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

load_dotenv(os.path.join(BACKEND_DIR, ".env"))

import psycopg
from psycopg.rows import dict_row

# Exact foreign-key dependency order: parents before children
MIGRATION_TABLE_ORDER = [
    "users",
    "issue_groups",
    "complaints",
    "feedback",
    "complaint_group_links",
    "menu_items",
    "issue_week_records",
    "response_cycles",
    "resolutions",
    "escalations",
    "risk_history",
    "ai_processing_jobs"
]

# Primary key column for conflict detection
TABLE_PRIMARY_KEYS = {
    "users": "id",
    "issue_groups": "id",
    "complaints": "id",
    "feedback": "id",
    "complaint_group_links": "complaint_id",
    "menu_items": "id",
    "issue_week_records": "id",
    "response_cycles": "id",
    "resolutions": "id",
    "escalations": "id",
    "risk_history": "id",
    "ai_processing_jobs": "id"
}

def ensure_sslmode(url: str) -> str:
    if not url:
        return url
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url

def get_sqlite_columns(c: sqlite3.Connection, table: str) -> List[str]:
    cur = c.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

def get_postgres_columns(pg_cur: psycopg.Cursor, table: str) -> List[str]:
    pg_cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s ORDER BY ordinal_position",
        (table,)
    )
    return [row["column_name"] for row in pg_cur.fetchall()]

def migrate_data(sqlite_path: str, pg_url: str, dry_run: bool = True) -> bool:
    if not os.path.exists(sqlite_path):
        print(f"ERROR: SQLite database file not found at: {sqlite_path}")
        return False

    print("=" * 70)
    print("MESSMATE: SQLITE TO POSTGRESQL DATA MIGRATION")
    print(f"Mode: {'DRY RUN (no changes will be committed)' if dry_run else 'LIVE MIGRATION'}")
    print(f"Source: {sqlite_path} (READ-ONLY)")
    print("=" * 70)

    # 1. Connect to SQLite strictly READ-ONLY
    sqlite_uri = f"file:{os.path.abspath(sqlite_path)}?mode=ro"
    sqlite_conn = sqlite3.connect(sqlite_uri, uri=True)
    sqlite_conn.row_factory = sqlite3.Row

    # 2. Connect to PostgreSQL
    normalized_pg_url = ensure_sslmode(pg_url)
    try:
        pg_conn = psycopg.connect(normalized_pg_url, autocommit=False, row_factory=dict_row)
    except Exception as e:
        print(f"ERROR: Could not connect to PostgreSQL destination: {e}")
        return False

    report = []
    has_errors = False

    try:
        with pg_conn.cursor() as pg_cur:
            # Verify destination database identity
            pg_cur.execute("SELECT current_database(), current_user, version()")
            dest_info = pg_cur.fetchone()
            print(f"Connected to PostgreSQL:")
            print(f"  Database : {dest_info['current_database']}")
            print(f"  User     : {dest_info['current_user']}")
            print(f"  Version  : {dest_info['version'].split(',')[0]}")
            print("-" * 70)

            # Discover available tables in SQLite
            sqlite_cur = sqlite_conn.cursor()
            sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            sqlite_tables = {row[0] for row in sqlite_cur.fetchall()}

            for table in MIGRATION_TABLE_ORDER:
                if table not in sqlite_tables:
                    report.append({
                        "table": table, "source_count": 0, "pre_count": 0,
                        "migrated": 0, "post_count": 0, "status": "SKIPPED (not in source)"
                    })
                    continue

                # Inspect source and destination columns
                sqlite_cols = get_sqlite_columns(sqlite_conn, table)
                pg_cols = get_postgres_columns(pg_cur, table)
                if not pg_cols:
                    print(f"WARNING: Table '{table}' does not exist in PostgreSQL. Run init_db() first.")
                    report.append({
                        "table": table, "source_count": 0, "pre_count": 0,
                        "migrated": 0, "post_count": 0, "status": "MISSING IN PG"
                    })
                    has_errors = True
                    continue

                # Match valid common columns
                common_cols = [c for c in sqlite_cols if c in pg_cols]
                pk_col = TABLE_PRIMARY_KEYS.get(table, "id")

                # Get row counts
                sqlite_cur.execute(f"SELECT COUNT(*) FROM {table}")
                src_count = sqlite_cur.fetchone()[0]

                pg_cur.execute(f"SELECT COUNT(*) as n FROM {table}")
                pre_count = pg_cur.fetchone()["n"]

                if src_count == 0:
                    report.append({
                        "table": table, "source_count": 0, "pre_count": pre_count,
                        "migrated": 0, "post_count": pre_count, "status": "EMPTY (0 rows)"
                    })
                    continue

                # Fetch all rows from SQLite
                sqlite_cur.execute(f"SELECT {', '.join(common_cols)} FROM {table} ORDER BY {pk_col} ASC")
                rows = sqlite_cur.fetchall()

                # Build parameterized PostgreSQL INSERT with ON CONFLICT DO NOTHING
                placeholders = ", ".join(["%s"] * len(common_cols))
                insert_sql = f"""
                    INSERT INTO {table} ({', '.join(common_cols)})
                    VALUES ({placeholders})
                    ON CONFLICT ({pk_col}) DO NOTHING
                """

                migrated_count = 0
                for r in rows:
                    vals = tuple(r[col] for col in common_cols)
                    pg_cur.execute(insert_sql, vals)
                    if pg_cur.rowcount > 0:
                        migrated_count += 1

                # Synchronize PostgreSQL sequence counter if table has 'id' SERIAL column
                if "id" in common_cols and pk_col == "id":
                    pg_cur.execute(f"SELECT setval(pg_get_serial_sequence(%s, 'id'), COALESCE(MAX(id), 1)) FROM {table}", (table,))

                # Verify post-migration count
                pg_cur.execute(f"SELECT COUNT(*) as n FROM {table}")
                post_count = pg_cur.fetchone()["n"]

                status = f"MIGRATED (+{migrated_count})" if migrated_count > 0 else "NO CHANGES (already present)"
                report.append({
                    "table": table, "source_count": src_count, "pre_count": pre_count,
                    "migrated": migrated_count, "post_count": post_count, "status": status
                })

            if dry_run:
                print("\n[DRY RUN] Rolling back all changes. No data was written to PostgreSQL.")
                pg_conn.rollback()
            else:
                print("\n[LIVE MIGRATION] Committing all transactions...")
                pg_conn.commit()
                print("Commit successful!")

    except Exception as e:
        pg_conn.rollback()
        print(f"\nFATAL ERROR DURING MIGRATION: {e}")
        has_errors = True
    finally:
        sqlite_conn.close()
        pg_conn.close()

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'TABLE NAME':<24} {'SQLITE':<8} {'PG PRE':<8} {'COPIED':<8} {'PG POST':<8} {'STATUS'}")
    print("-" * 80)
    for r in report:
        print(f"{r['table']:<24} {r['source_count']:<8} {r['pre_count']:<8} {r['migrated']:<8} {r['post_count']:<8} {r['status']}")
    print("=" * 80)

    return not has_errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MessMate SQLite to PostgreSQL Migration")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Inspect and simulate migration without committing changes")
    parser.add_argument("--confirm", action="store_true", default=False, help="Explicit confirmation required to execute live migration")
    parser.add_argument("--sqlite-path", default=os.path.join(BACKEND_DIR, "messmate.db"), help="Path to source SQLite database")
    parser.add_argument("--postgres-url", default=os.getenv("DATABASE_URL"), help="Destination PostgreSQL connection URL")

    args = parser.parse_args()

    if not args.postgres_url:
        print("ERROR: DATABASE_URL is not set in environment or passed via --postgres-url.")
        sys.exit(1)

    if not args.confirm and not args.dry_run:
        print("SAFETY BLOCK: You must pass either --dry-run to simulate, or --confirm to execute.")
        print("Example: python scripts/migrate_sqlite_to_postgres.py --dry-run")
        sys.exit(1)

    success = migrate_data(
        sqlite_path=args.sqlite_path,
        pg_url=args.postgres_url,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)
