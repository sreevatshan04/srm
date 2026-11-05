"""
MessMate Risk-Score Engine
Deterministic, backend-controlled risk scoring for college mess feedback.

Formulas:
  R_raw = W + D
  R_current = max(0, min(100, W + D - C))

Where:
  W = min(45, 15 * (N - 1))  # N = distinct calendar weeks in Asia/Kolkata
  D = min(30, 10 + 5 * floor(T / 24)) after 24h deadline missed; 0 before deadline
  C = sum of verified resolution credits (default 15 per verified event)

Risk Levels:
  0 - 29:   LOW
  30 - 59:  MEDIUM
  60 - 79:  HIGH (triggers inspection escalation)
  80 - 100: CRITICAL (urgent escalation)
"""

import os
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, Tuple, List

# Timezone for calendar week boundary determination
DEFAULT_TZ_NAME = os.getenv("APP_TIMEZONE", "Asia/Kolkata")
try:
    TZ = ZoneInfo(DEFAULT_TZ_NAME)
except Exception:
    TZ = timezone(timedelta(hours=5, minutes=30))

RESOLUTION_REDUCTION_POINTS = int(os.getenv("RESOLUTION_REDUCTION_POINTS", "15"))
ESCALATION_THRESHOLD = 60

def get_calendar_week_key(dt: datetime) -> str:
    """
    Returns an ISO calendar week key string, e.g. '2026-W43', based on Asia/Kolkata.
    Weeks run Monday through Sunday.
    """
    if dt.tzinfo is None:
        # Assume UTC if naive, convert to college timezone
        dt = dt.replace(tzinfo=timezone.utc).astimezone(TZ)
    else:
        dt = dt.astimezone(TZ)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"

def calculate_recurrence_points(distinct_weeks_count: int) -> int:
    """
    W = min(45, 15 * (N - 1)) for N >= 1; 0 if N <= 1.
    1 week  -> 0
    2 weeks -> 15
    3 weeks -> 30
    4+ weeks -> 45 (cap)
    """
    if distinct_weeks_count <= 1:
        return 0
    return min(45, 15 * (distinct_weeks_count - 1))

def calculate_delay_points(
    started_at: datetime,
    responded_at: Optional[datetime] = None,
    now_dt: Optional[datetime] = None,
    deadline_hours: int = 24
) -> Tuple[int, bool, float]:
    """
    Calculates delay penalty D according to:
      - Within 24h: D = 0
      - Overdue: D = min(30, 10 + 5 * floor(T / 24))
        where T is complete hours elapsed since deadline.
    Returns: (delay_points, is_overdue, hours_overdue)
    """
    # Normalize to UTC
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    
    deadline = started_at + timedelta(hours=deadline_hours)
    
    # Effective check time: if responded, delay freezes at responded_at
    if responded_at is not None:
        if responded_at.tzinfo is None:
            responded_at = responded_at.replace(tzinfo=timezone.utc)
        effective_time = responded_at
    else:
        now_dt = now_dt or datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        effective_time = now_dt

    if effective_time <= deadline:
        return 0, False, 0.0

    # Deadline was missed
    overdue_seconds = (effective_time - deadline).total_seconds()
    hours_overdue = overdue_seconds / 3600.0
    complete_hours_since_deadline = math.floor(hours_overdue)
    
    # 10 pts immediate + 5 pts per additional 24 complete hours
    additional_24h_periods = complete_hours_since_deadline // 24
    delay_points = min(30, 10 + (5 * additional_24h_periods))
    
    return delay_points, True, round(hours_overdue, 2)

def calculate_risk_score(
    recurrence_points: int,
    delay_points: int,
    resolution_credits: int = 0
) -> Dict[str, Any]:
    """
    Calculates final risk score:
      R_current = max(0, min(100, W + D - C))
    Returns detailed breakdown and risk level.
    """
    raw_score = recurrence_points + delay_points
    adjusted_score = max(0, min(100, raw_score - resolution_credits))
    level = determine_risk_level(adjusted_score)
    
    return {
        "score": adjusted_score,
        "raw_score": raw_score,
        "level": level,
        "recurrence_points": recurrence_points,
        "delay_points": delay_points,
        "resolution_credits": resolution_credits,
        "breakdown": {
            "recurrence": recurrence_points,
            "delay": delay_points,
            "credits": -resolution_credits,
            "total": adjusted_score
        }
    }

def determine_risk_level(score: int) -> str:
    """
    0 - 29:   LOW
    30 - 59:  MEDIUM
    60 - 79:  HIGH
    80 - 100: CRITICAL
    """
    if score >= 80:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    return "LOW"

def should_escalate(score: int, is_safety_hazard: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Returns (should_escalate, reason)
    - If is_safety_hazard is True -> urgent_food_safety
    - If score >= 60 -> threshold_reached
    """
    if is_safety_hazard:
        return True, "urgent_food_safety"
    if score >= ESCALATION_THRESHOLD:
        return True, "threshold_reached"
    return False, None

def reconcile_issue_risk(c, group_id: int, now_dt: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Authoritative server-side risk recalculation for an issue group.
    Updates PostgreSQL state atomically:
      1. Counts distinct calendar weeks from issue_week_records
      2. Calculates recurrence points W
      3. Determines delay penalty D from active response cycle
      4. Calculates verified resolution credits C
      5. Computes R = max(0, min(100, W + D - C))
      6. Evaluates and creates escalation if eligible (idempotently)
      7. Records score change in risk_history
    """
    now_dt = now_dt or datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    row = c.execute("SELECT * FROM issue_groups WHERE id=%s", (group_id,)).fetchone()
    if not row:
        return {}

    issue = dict(row)
    
    # 1. Distinct calendar weeks (W)
    week_count = c.execute(
        "SELECT COUNT(DISTINCT week_key) as n FROM issue_week_records WHERE group_id=%s",
        (group_id,)
    ).fetchone()["n"]
    if week_count == 0:
        # Fallback to service_date if no week record yet
        try:
            s_dt = datetime.fromisoformat(issue["service_date"])
            w_key = get_calendar_week_key(s_dt)
            c.execute(
                """INSERT INTO issue_week_records(group_id, week_key, first_seen_at)
                   VALUES(%s,%s,%s)
                   ON CONFLICT (group_id, week_key) DO NOTHING""",
                (group_id, w_key, issue.get("first_reported_at") or now_iso)
            )
            week_count = 1
        except Exception:
            week_count = 1

    recurrence_pts = calculate_recurrence_points(week_count)

    # 2. Response cycle delay (D)
    # Fetch active or latest cycle
    cycle_row = c.execute(
        "SELECT * FROM response_cycles WHERE group_id=%s ORDER BY cycle_number DESC LIMIT 1",
        (group_id,)
    ).fetchone()
    
    if cycle_row:
        cycle = dict(cycle_row)
        started_at = datetime.fromisoformat(cycle["started_at"])
        responded_at = datetime.fromisoformat(cycle["responded_at"]) if cycle["responded_at"] else None
        delay_pts, is_overdue, hours_overdue = calculate_delay_points(
            started_at=started_at,
            responded_at=responded_at,
            now_dt=now_dt
        )
        # Update cycle status if needed
        new_cycle_status = "Responded" if responded_at else ("Overdue" if is_overdue else "Open")
        if new_cycle_status != cycle["status"]:
            c.execute("UPDATE response_cycles SET status=%s WHERE id=%s", (new_cycle_status, cycle["id"]))
    else:
        # Synthesize from issue_groups deadline / first_reported_at
        deadline_str = issue.get("deadline")
        started_str = issue.get("first_reported_at")
        responded_str = issue.get("responded_at")
        
        try:
            deadline_dt = datetime.fromisoformat(deadline_str) if deadline_str else None
        except Exception:
            deadline_dt = None
            
        try:
            started_at = datetime.fromisoformat(started_str) if started_str else (deadline_dt - timedelta(hours=24) if deadline_dt else now_dt)
        except Exception:
            started_at = now_dt
            
        responded_at = None
        if responded_str:
            try:
                responded_at = datetime.fromisoformat(responded_str)
            except Exception:
                pass

        delay_pts, is_overdue, hours_overdue = calculate_delay_points(
            started_at=started_at,
            responded_at=responded_at,
            now_dt=now_dt
        )

    # 3. Verified Resolution Credits (C)
    credits_sum = c.execute(
        "SELECT COALESCE(SUM(credit_applied), 0) as s FROM resolutions WHERE group_id=%s",
        (group_id,)
    ).fetchone()["s"]

    # 4. Final Risk Calculation
    risk_data = calculate_risk_score(
        recurrence_points=recurrence_pts,
        delay_points=delay_pts,
        resolution_credits=credits_sum
    )
    final_score = risk_data["score"]
    risk_level = risk_data["level"]

    old_score = issue.get("risk_score") or 0
    is_safety_hazard = bool(issue.get("is_food_safety_hazard", 0))

    # 5. Update issue_groups
    c.execute(
        """UPDATE issue_groups 
           SET risk_score=%s, risk_level=%s, recurrence_score=%s, delay_score=%s, 
               resolution_credits=%s, distinct_weeks=%s
           WHERE id=%s""",
        (final_score, risk_level, recurrence_pts, delay_pts, credits_sum, week_count, group_id)
    )

    # 6. Audit in risk_history if score changed
    if old_score != final_score:
        import json
        c.execute(
            """INSERT INTO risk_history(group_id, old_score, new_score, reason, breakdown_json, created_at)
               VALUES(%s,%s,%s,%s,%s,%s)""",
            (group_id, old_score, final_score, "Score recalculated", json.dumps(risk_data["breakdown"]), now_iso)
        )

    # 7. Escalation Evaluation
    esc_needed, esc_reason = should_escalate(final_score, is_safety_hazard=is_safety_hazard)
    if esc_needed:
        # Ensure idempotent: only insert if no ACTIVE escalation exists
        active_esc = c.execute(
            "SELECT id FROM escalations WHERE group_id=%s AND status='Active'",
            (group_id,)
        ).fetchone()
        if not active_esc:
            c.execute(
                """INSERT INTO escalations(group_id, reason, score_at_trigger, triggered_at, status)
                   VALUES(%s,%s,%s,%s, 'Active')""",
                (group_id, esc_reason, final_score, now_iso)
            )

    return risk_data

def evaluate_positive_feedback_trigger_b(c, group_id: int, now_dt: Optional[datetime] = None) -> bool:
    """
    Trigger B: Positive student feedback in subsequent weeks.
    Threshold: Average rating >= 4.0 with at least 3 ratings in a calendar week
    occurring after the issue's initial report week.
    Grants 15 points credit once per qualifying cycle.
    """
    now_dt = now_dt or datetime.now(timezone.utc)
    issue = c.execute("SELECT * FROM issue_groups WHERE id=%s", (group_id,)).fetchone()
    if not issue:
        return False
        
    # Check if already received a Trigger B credit
    existing_b = c.execute(
        "SELECT id FROM resolutions WHERE group_id=%s AND trigger_type='positive_feedback'",
        (group_id,)
    ).fetchone()
    if existing_b:
        return False

    meal_id = issue["meal_id"]
    try:
        first_dt = datetime.fromisoformat(issue["first_reported_at"] or issue["service_date"])
        initial_week = get_calendar_week_key(first_dt)
    except Exception:
        return False

    # Check feedback ratings submitted in subsequent weeks
    ratings_by_week = c.execute(
        "SELECT service_date, rating FROM feedback WHERE meal_id=%s",
        (meal_id,)
    ).fetchall()

    weeks_data = {}
    for r in ratings_by_week:
        try:
            d = datetime.fromisoformat(r["service_date"])
            w = get_calendar_week_key(d)
            if w > initial_week: # strictly subsequent week
                weeks_data.setdefault(w, []).append(r["rating"])
        except Exception:
            continue

    for w, r_list in weeks_data.items():
        if len(r_list) >= 3 and (sum(r_list) / len(r_list)) >= 4.0:
            # Condition met! Apply Trigger B resolution credit
            c.execute(
                """INSERT INTO resolutions(group_id, trigger_type, verified_by, credit_applied, notes, verified_at)
                   VALUES(%s, 'positive_feedback', 'Automated Feedback Monitor', %s, %s, %s)""",
                (group_id, RESOLUTION_REDUCTION_POINTS, f"Subsequent week {w} average rating exceeded 4.0 ({len(r_list)} reviews)", now_dt.isoformat())
            )
            reconcile_issue_risk(c, group_id, now_dt=now_dt)
            return True
    return False

def reconcile_all_issues(connect_fn, now_dt: Optional[datetime] = None):
    """
    Periodic / on-demand reconciliation of all active issues.
    """
    now_dt = now_dt or datetime.now(timezone.utc)
    with connect_fn() as c:
        issue_ids = [r["id"] for r in c.execute("SELECT id FROM issue_groups").fetchall()]
        for gid in issue_ids:
            evaluate_positive_feedback_trigger_b(c, gid, now_dt=now_dt)
            reconcile_issue_risk(c, gid, now_dt=now_dt)
        c.commit()
