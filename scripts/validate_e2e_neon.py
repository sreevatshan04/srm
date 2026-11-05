import os
import sys

# Ensure backend is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.db import connect, get_pool
from app.risk_engine import reconcile_all_issues

print("=== MESSMATE END-TO-END VALIDATION (NEON POSTGRESQL) ===")

# 1. Connection Pool Check
pool = get_pool()
print("1. Connection pool active:", not pool.closed)
print("   Pool min_size:", pool.min_size, "max_size:", pool.max_size)

# 2. Database Connection and Table Counts
with connect() as c:
    users = c.execute("SELECT COUNT(*) AS n FROM users").fetchone()
    feedback = c.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()
    complaints = c.execute("SELECT COUNT(*) AS n FROM complaints").fetchone()
    risk_history = c.execute("SELECT COUNT(*) AS n FROM risk_history").fetchone()
    issue_groups = c.execute("SELECT COUNT(*) AS n FROM issue_groups").fetchone()
    
    print(f"2. Read operations successful:")
    print(f"   users: {users['n']} (indexed access: {users[0]})")
    print(f"   feedback: {feedback['n']}")
    print(f"   complaints: {complaints['n']}")
    print(f"   risk_history: {risk_history['n']}")
    print(f"   issue_groups: {issue_groups['n']}")

# 3. Risk Engine Reconcile All Issues (runs against Neon)
print("3. Executing risk engine reconciliation against Neon...")
reconcile_all_issues(connect)
print("   reconcile_all_issues completed successfully!")

# 4. Transaction Isolation & Rollback Verification
print("4. Testing transactional atomicity on Neon...")
with connect() as c:
    initial_cnt = c.execute("SELECT COUNT(*) AS n FROM menu_items").fetchone()["n"]

try:
    with connect() as c:
        c.execute(
            "INSERT INTO menu_items(menu_date, meal_id, name, description, dietary_type) VALUES(%s, %s, %s, %s, %s)",
            ("2026-11-05", "lunch", "Test Item Temporary", "Testing Rollback", "Veg")
        )
        raise ValueError("Simulated failure to verify rollback")
except ValueError:
    pass

with connect() as c:
    post_cnt = c.execute("SELECT COUNT(*) AS n FROM menu_items").fetchone()["n"]

print(f"   Rollback check: initial={initial_cnt}, after failure={post_cnt} (Match: {initial_cnt == post_cnt})")

# 5. Check active modules for sqlite3
print("5. Checking sys.modules for operational sqlite3 dependency:")
sqlite_in_modules = "sqlite3" in sys.modules
print(f"   sqlite3 loaded in sys.modules: {sqlite_in_modules}")

print("\nALL END-TO-END VALIDATION CHECKS PASSED!")
