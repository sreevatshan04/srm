"""
Comprehensive Integration Tests for MessMate:
- PostgreSQL Schema Integrity (via ANSI information_schema)
- Risk Score Engine
- Response Cycles & Deadlines
- Resolution Verification & Credits
- Escalations & Food Safety Overrides
- Gemini Integration & Fallback
- Privacy & Role-Based Access Control
- Transaction Atomicity & Rollback
- Foreign Key Enforcement
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient

from app import main as main_module
from app.db import connect, IntegrityError
from app.risk_engine import (
    calculate_recurrence_points,
    calculate_delay_points,
    calculate_risk_score,
    reconcile_issue_risk,
    reconcile_all_issues,
    get_calendar_week_key
)
from app.gemini_service import (
    call_gemini_grouping,
    normalise_rule_based
)

os.environ["FIREBASE_MANAGER_EMAILS"] = "manager@test.invalid"
os.environ["FIREBASE_REVIEWER_EMAILS"] = "reviewer@test.invalid"
main_module.verify_firebase_token = lambda token: {
    "uid": f"test-{token}",
    "email": token,
    "name": token.split("@", 1)[0],
    "email_verified": True,
}
client = TestClient(main_module.app)

def auth_headers(email):
    return {"Authorization": f"Bearer {email}"}

def test_database_schema_integrity():
    with connect() as c:
        # Standard PostgreSQL information_schema query
        rows = c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
        tables = [r["table_name"] for r in rows]
        assert "users" in tables
        assert "complaints" in tables
        assert "feedback" in tables
        assert "issue_groups" in tables
        assert "issue_week_records" in tables
        assert "response_cycles" in tables
        assert "resolutions" in tables
        assert "escalations" in tables
        assert "risk_history" in tables

        # Verify columns on issue_groups
        col_rows = c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='issue_groups'"
        ).fetchall()
        cols = [r["column_name"] for r in col_rows]
        for expected in ["risk_score", "risk_level", "recurrence_score", "delay_score", "resolution_credits", "is_food_safety_hazard", "distinct_weeks", "category"]:
            assert expected in cols, f"Missing column {expected} in issue_groups"

def test_role_enforcement_and_access_control():
    std_headers = auth_headers("student@test.invalid")

    # Student trying to access manager endpoints -> 403 Forbidden
    r_forbidden = client.get("/api/manager/overview", headers=std_headers)
    assert r_forbidden.status_code == 403, "Student should not access manager overview"

    r_issues_forbidden = client.get("/api/manager/issues", headers=std_headers)
    assert r_issues_forbidden.status_code == 403

    mgr_headers = auth_headers("manager@test.invalid")

    # Manager can access manager overview
    r_mgr_over = client.get("/api/manager/overview", headers=mgr_headers)
    assert r_mgr_over.status_code == 200

    rev_headers = auth_headers("reviewer@test.invalid")

    r_rev_esc = client.get("/api/manager/escalations", headers=rev_headers)
    assert r_rev_esc.status_code == 200

def test_student_privacy_guarantee():
    mgr_headers = auth_headers("manager@test.invalid")

    # 1. Check overview endpoint
    over_data = client.get("/api/manager/overview", headers=mgr_headers).json()
    serialized = json.dumps(over_data)
    assert "student@srm.edu" not in serialized
    assert "RA2311003010001" not in serialized
    assert "Vatsan" not in serialized

    # 2. Check issues endpoint
    issues_data = client.get("/api/manager/issues", headers=mgr_headers).json()
    serialized_issues = json.dumps(issues_data)
    assert "student@srm.edu" not in serialized_issues
    assert "RA2311003010001" not in serialized_issues

    # 3. Check issue details endpoint
    if issues_data["issues"]:
        first_id = issues_data["issues"][0]["id"]
        detail_data = client.get(f"/api/manager/issues/{first_id}", headers=mgr_headers).json()
        serialized_detail = json.dumps(detail_data)
        assert "student@srm.edu" not in serialized_detail
        assert "RA2311003010001" not in serialized_detail

def test_risk_score_calculation_and_escalation_flow():
    # Clean up any leftover test data
    with connect() as c:
        c.execute("DELETE FROM issue_groups WHERE issue_key = 'test_curry_salt'")
        c.commit()

    now_iso = datetime.now(timezone.utc).isoformat()
    past_started = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat() # 50h ago -> 26h overdue -> 10 pts delay
    deadline_iso = (datetime.now(timezone.utc) - timedelta(hours=26)).isoformat()

    with connect() as c:
        cur = c.execute(
            """INSERT INTO issue_groups(meal_id, service_date, issue_key, category, title, summary, count, deadline, status, first_reported_at)
               VALUES('lunch', '2026-10-01', 'test_curry_salt', 'Food taste', 'Salt Test Issue', 'Too much salt in curry', 5, %s, 'Open', %s) RETURNING id""",
            (deadline_iso, past_started)
        )
        group_id = cur.fetchone()["id"]
        # Add 3 distinct weeks: Week 40, Week 41, Week 42 -> 3 distinct weeks -> W = 30
        c.execute("INSERT INTO issue_week_records(group_id, week_key, first_seen_at) VALUES(%s, '2026-W40', %s) ON CONFLICT (group_id, week_key) DO NOTHING", (group_id, past_started))
        c.execute("INSERT INTO issue_week_records(group_id, week_key, first_seen_at) VALUES(%s, '2026-W41', %s) ON CONFLICT (group_id, week_key) DO NOTHING", (group_id, past_started))
        c.execute("INSERT INTO issue_week_records(group_id, week_key, first_seen_at) VALUES(%s, '2026-W42', %s) ON CONFLICT (group_id, week_key) DO NOTHING", (group_id, past_started))

        # Add active response cycle
        c.execute(
            """INSERT INTO response_cycles(group_id, cycle_number, started_at, deadline, status)
               VALUES(%s, 1, %s, %s, 'Overdue') ON CONFLICT (group_id, cycle_number) DO NOTHING""",
            (group_id, past_started, deadline_iso)
        )
        c.commit()

        # Reconcile risk
        reconcile_issue_risk(c, group_id)
        c.commit()

        row = c.execute("SELECT * FROM issue_groups WHERE id=%s", (group_id,)).fetchone()
        # W = 30 (3 weeks), D = 15 (26h overdue = initial 10 + 5*(26//24) = 15), C = 0 -> R = 45 -> MEDIUM
        assert row["recurrence_score"] == 30
        assert row["delay_score"] == 15
        assert row["risk_score"] == 45
        assert row["risk_level"] == "MEDIUM"

        # Now add Week 43 and Week 44 -> 5 distinct weeks -> W capped at 45
        # Total raw score = 45 + 15 = 60 -> HIGH -> ESCALATION TRIGGERED!
        c.execute("INSERT INTO issue_week_records(group_id, week_key, first_seen_at) VALUES(%s, '2026-W43', %s) ON CONFLICT (group_id, week_key) DO NOTHING", (group_id, past_started))
        c.execute("INSERT INTO issue_week_records(group_id, week_key, first_seen_at) VALUES(%s, '2026-W44', %s) ON CONFLICT (group_id, week_key) DO NOTHING", (group_id, past_started))
        c.commit()

        reconcile_issue_risk(c, group_id)
        c.commit()

        row_escalated = c.execute("SELECT * FROM issue_groups WHERE id=%s", (group_id,)).fetchone()
        assert row_escalated["recurrence_score"] == 45
        assert row_escalated["risk_score"] == 60
        assert row_escalated["risk_level"] == "HIGH"

        # Verify active escalation record was created automatically
        esc = c.execute("SELECT * FROM escalations WHERE group_id=%s AND status='Active'", (group_id,)).fetchone()
        assert esc is not None
        assert esc["reason"] == "threshold_reached"
        assert esc["score_at_trigger"] == 60
        esc_id = esc["id"]

        # Verify idempotence: reconciling again does NOT create duplicate active escalation
        reconcile_issue_risk(c, group_id)
        c.commit()
        active_count = c.execute("SELECT COUNT(*) n FROM escalations WHERE group_id=%s AND status='Active'", (group_id,)).fetchone()["n"]
        assert active_count == 1, "Escalation must be idempotent"

    # Manager responds -> stops delay penalty
    mgr_headers = auth_headers("manager@test.invalid")

    r_resp = client.post(f"/api/manager/issues/{group_id}/respond", headers=mgr_headers, json={
        "response": "Reduced salt in standard recipe and retrained cook staff."
    })
    assert r_resp.status_code == 200

    with connect() as c:
        row_after_resp = c.execute("SELECT * FROM issue_groups WHERE id=%s", (group_id,)).fetchone()
        assert row_after_resp["status"] == "Responded"
        # Delay is frozen, recurrence history is PRESERVED
        assert row_after_resp["recurrence_score"] == 45

    # Reviewer verifies resolution -> Trigger A resolution credit (-15 pts)
    rev_headers = auth_headers("reviewer@test.invalid")

    r_verify = client.post(f"/api/manager/issues/{group_id}/verify-resolution", headers=rev_headers, json={
        "notes": "Inspected kitchen, verified salt levels within campus nutrition standards.",
        "trigger_type": "corrective_action"
    })
    assert r_verify.status_code == 200
    verify_data = r_verify.json()
    assert verify_data["credit_applied"] == 15
    # Score was 60, now 60 - 15 = 45 (MEDIUM)
    assert verify_data["new_score"] == 45
    assert verify_data["new_level"] == "MEDIUM"

    # Duplicate verification in same state rejected
    r_dup = client.post(f"/api/manager/issues/{group_id}/verify-resolution", headers=rev_headers, json={
        "notes": "Attempting duplicate credit",
        "trigger_type": "corrective_action"
    })
    assert r_dup.status_code == 400, "Duplicate resolution credit must be blocked"

    # Reviewer closes inspection escalation
    r_close = client.post(f"/api/manager/escalations/{esc_id}/close", headers=rev_headers, json={
        "closure_notes": "Inspection completed successfully; kitchen recipe updated."
    })
    assert r_close.status_code == 200

    with connect() as c:
        esc_closed = c.execute("SELECT * FROM escalations WHERE id=%s", (esc_id,)).fetchone()
        assert esc_closed["status"] == "Closed"
        assert esc_closed["closure_notes"] is not None

def test_urgent_food_safety_override():
    # Clean up any leftover test data
    with connect() as c:
        c.execute("DELETE FROM issue_groups WHERE issue_key = 'safety_override_test'")
        c.commit()

    rev_headers = auth_headers("reviewer@test.invalid")

    now_iso = datetime.now(timezone.utc).isoformat()
    with connect() as c:
        cur = c.execute(
            """INSERT INTO issue_groups(meal_id, service_date, issue_key, category, title, summary, count, deadline, status)
               VALUES('dinner', '2026-10-24', 'safety_override_test', 'Hygiene and cleanliness', 'Safety Test', 'Insect reported', 1, %s, 'Open') RETURNING id""",
            (now_iso,)
        )
        gid = cur.fetchone()["id"]
        c.commit()

    # Trigger urgent food safety override
    r_ovr = client.post(f"/api/manager/issues/{gid}/override-safety", headers=rev_headers, json={
        "is_safety_hazard": True,
        "reason": "Insect reported in food"
    })
    assert r_ovr.status_code == 200

    # Even though numerical score is low, an urgent food safety escalation was created!
    with connect() as c:
        esc = c.execute("SELECT * FROM escalations WHERE group_id=%s AND status='Active'", (gid,)).fetchone()
        assert esc is not None
        assert esc["reason"] == "urgent_food_safety"

def test_gemini_fallback_and_validation():
    slug, cat, title, summary, conf, is_haz = normalise_rule_based("Sambar was completely cold today")
    assert cat == "Food temperature"
    assert not is_haz
    assert conf >= 0.8

    slug2, cat2, title2, summary2, conf2, is_haz2 = normalise_rule_based("Found a dead cockroach in the curry")
    assert is_haz2 is True
    assert cat2 == "Hygiene and cleanliness"

def test_foreign_key_enforcement():
    """Verify that PostgreSQL foreign key constraints block invalid relational inserts."""
    with connect() as c:
        import pytest
        with pytest.raises(Exception):
            # Attempt to insert feedback referencing non-existent user_id 999999
            c.execute(
                """INSERT INTO feedback(user_id, meal_id, service_date, rating, items_json, comment, category, dietary, created_at)
                   VALUES(999999, 'lunch', '2026-11-05', 5, '{}', 'test', 'General', 'veg', '2026-11-05T12:00:00')"""
            )
            c.commit()

def test_transaction_rollback_on_failure():
    """Verify that a transaction error rolls back changes atomically."""
    test_identifier = "rollback_test_user@srm.edu"
    try:
        with connect() as c:
            c.execute(
                "INSERT INTO users(identifier, name, role, password_hash) VALUES(%s, 'Rollback Test', 'student', 'hash')",
                (test_identifier,)
            )
            # Intentionally cause a syntax or constraint error in the same transaction
            c.execute("INSERT INTO non_existent_table_for_rollback_test VALUES(1)")
            c.commit()
    except Exception:
        pass

    with connect() as c:
        user = c.execute("SELECT id FROM users WHERE identifier=%s", (test_identifier,)).fetchone()
        assert user is None, "Failed transaction must roll back uncommitted inserts"
