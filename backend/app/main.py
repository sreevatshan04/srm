from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import os, secrets, hashlib, hmac, json, re
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

# Load environment variables (.env in backend or root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from app.db import connect, init_db, IntegrityError
from app.menus import OFFICIAL_MENUS, get_official_menu, WEEKDAYS
from app.risk_engine import (
    calculate_recurrence_points,
    calculate_delay_points,
    calculate_risk_score,
    determine_risk_level,
    should_escalate,
    get_calendar_week_key,
    reconcile_issue_risk,
    evaluate_positive_feedback_trigger_b,
    reconcile_all_issues,
    RESOLUTION_REDUCTION_POINTS,
    ESCALATION_THRESHOLD
)
from app.gemini_service import (
    process_pending_complaints_job,
    get_ai_status,
    normalise_rule_based,
    generate_meal_summary
)

TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Kolkata"))
ALLOW_FEEDBACK_OUTSIDE_WINDOW = os.getenv(
    "ALLOW_FEEDBACK_OUTSIDE_WINDOW",
    "false" if os.getenv("VERCEL") or os.getenv("APP_ENV", "").lower() == "production" else "true",
).lower() in {"1", "true", "yes"}
ALLOW_MULTIPLE_FEEDBACK = os.getenv(
    "ALLOW_MULTIPLE_FEEDBACK",
    "false" if os.getenv("VERCEL") or os.getenv("APP_ENV", "").lower() == "production" else "true",
).lower() in {"1", "true", "yes"}

app = FastAPI(title="MessMate API", version="2.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MEALS = [
    {"id": "breakfast", "name": "Breakfast", "start": "07:00", "end": "11:00", "time": "7:00 AM – 11:00 AM"},
    {"id": "lunch", "name": "Lunch", "start": "12:30", "end": "15:00", "time": "12:30 PM – 3:00 PM"},
    {"id": "snacks", "name": "Snacks", "start": "15:00", "end": "17:00", "time": "3:00 PM – 5:00 PM"},
    {"id": "dinner", "name": "Dinner", "start": "19:00", "end": "22:00", "time": "7:00 PM – 10:00 PM"},
]

def menu_for_date(day, dietary="non_veg"):
    day_name = day.strftime("%A").lower()
    return {m: get_official_menu(day_name, m, dietary) for m in ["breakfast", "lunch", "snacks", "dinner"]}

def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 180000).hex()
    return f"{salt}${hashed}"

def verify_firebase_token(token: str) -> dict:
    """Verify Firebase ID tokens with the Admin SDK; never trust client claims."""
    try:
        firebase_admin.get_app()
    except ValueError:
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("FIREBASE_PROJECT_ID", "mess-mate-418df")
        if service_account_json:
            credential = credentials.Certificate(json.loads(service_account_json))
        elif service_account_path:
            credential = credentials.Certificate(service_account_path)
        else:
            credential = credentials.ApplicationDefault()
        firebase_admin.initialize_app(credential, {"projectId": project_id})
    try:
        claims = firebase_auth.verify_id_token(token, check_revoked=True)
    except Exception as error:
        raise HTTPException(401, "Firebase session is invalid or expired. Please sign in again.") from error
    if claims.get("aud") != os.getenv("FIREBASE_PROJECT_ID", "mess-mate-418df"):
        raise HTTPException(401, "Firebase token belongs to a different project.")
    return claims

def firebase_profile(claims: dict) -> dict:
    uid = claims.get("uid")
    email = (claims.get("email") or "").strip().lower()
    if not uid or not email:
        raise HTTPException(403, "A Firebase account with an email address is required.")
    if claims.get("email_verified") is not True:
        raise HTTPException(403, "Verify your email address before using MessMate.")
    manager_emails = {item.strip().lower() for item in os.getenv("FIREBASE_MANAGER_EMAILS", "").split(",") if item.strip()}
    reviewer_emails = {item.strip().lower() for item in os.getenv("FIREBASE_REVIEWER_EMAILS", "").split(",") if item.strip()}
    role = "reviewer" if email in reviewer_emails else "manager" if email in manager_emails else "student"
    name = str(claims.get("name") or email.split("@", 1)[0])[:120]

    with connect() as connection:
        user = connection.execute("SELECT * FROM users WHERE firebase_uid=%s OR LOWER(identifier)=%s", (uid, email)).fetchone()
        if user:
            if user["firebase_uid"] and user["firebase_uid"] != uid:
                raise HTTPException(409, "This email is already linked to another Firebase account.")
            connection.execute(
                "UPDATE users SET firebase_uid=%s, identifier=%s, name=%s, role=%s WHERE id=%s",
                (uid, email, name, role, user["id"]),
            )
            user_id = user["id"]
        else:
            cursor = connection.execute(
                """INSERT INTO users(firebase_uid,identifier,name,role,password_hash)
                   VALUES(%s,%s,%s,%s,%s) RETURNING id""",
                (uid, email, name, role, hash_password(secrets.token_urlsafe(64))),
            )
            user_id = cursor.fetchone()["id"]
        connection.commit()
        user = connection.execute("SELECT * FROM users WHERE id=%s", (user_id,)).fetchone()
    return {
        "id": user["id"], "name": user["name"], "role": user["role"],
        "identifier": user["identifier"], "register_no": user["register_no"],
        "department": user["department"], "semester": user["semester"],
        "theme": user["theme"], "dietary_preference": user["dietary_preference"],
    }

# Startup database initialization and seeding
def _seed_initial_groups():
    try:
        with connect() as c:
            unprocessed = c.execute("SELECT COUNT(*) n FROM complaints WHERE processed_for_ai=0").fetchone()["n"]
        if unprocessed > 0:
            process_pending_complaints_job(connect, {m["id"]: m for m in MEALS}, TZ)
        reconcile_all_issues(connect)
    except Exception as e:
        # Non-fatal during static builds, offline tooling, or unconfigured testing
        print(f"[MessMate] Database seeding deferred: {e}")

try:
    init_db()
    _seed_initial_groups()
except Exception as e:
    print(f"[MessMate] Database initialization deferred: {e}")

@app.post("/api/auth/firebase")
def login_with_firebase(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "A Firebase ID token is required.")
    claims = verify_firebase_token(authorization.split(" ", 1)[1])
    user = firebase_profile(claims)
    return {"token": authorization.split(" ", 1)[1], "user": user}

class FeedbackBody(BaseModel):
    meal_id: str
    service_date: str
    rating: int = Field(ge=1, le=5)
    items: dict = {}
    comment: str = ""
    category: str = ""
    dietary: str = "non_veg"

class ComplaintBody(BaseModel):
    meal_id: str
    service_date: str
    category: str
    title: str
    description: str

class ResponseBody(BaseModel):
    response: str

class ResolutionBody(BaseModel):
    notes: str = ""
    trigger_type: str = "corrective_action"
    credit_amount: Optional[int] = None

class EscalationCloseBody(BaseModel):
    closure_notes: str

class SafetyOverrideBody(BaseModel):
    is_safety_hazard: bool
    reason: str = ""

class ThemeBody(BaseModel):
    theme: str

class MenuItemBody(BaseModel):
    menu_date: str
    meal_id: str
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    dietary_type: str = "Veg"
    id: Optional[int] = None

def current_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Please sign in to continue")
    token = authorization.split(" ", 1)[1]
    claims = verify_firebase_token(token)
    return firebase_profile(claims)

def require_role(role):
    def dependency(user=Depends(current_user)):
        if user["role"] != role:
            raise HTTPException(403, "You do not have permission to access this page")
        return user
    return dependency

def require_roles(*roles):
    def dependency(user=Depends(current_user)):
        if user["role"] not in roles:
            raise HTTPException(403, "You do not have permission to access this action")
        return user
    return dependency

def active_meal(now=None):
    now = now or datetime.now(TZ)
    hm = now.strftime("%H:%M")
    for m in MEALS:
        if m["start"] <= hm < m["end"]:
            return m
    return None

def meal_by_id(meal_id):
    for m in MEALS:
        if m["id"] == meal_id:
            return m
    raise HTTPException(404, "Invalid meal")

def resolve_menu_items(target_date: date, meal_id: str, dietary: str = "non_veg") -> List[str]:
    d_str = target_date.isoformat()
    with connect() as c:
        dtype_filter = "Veg" if dietary == "veg" else None
        if dtype_filter:
            rows = c.execute("SELECT name FROM menu_items WHERE menu_date=%s AND meal_id=%s AND dietary_type='Veg' ORDER BY id", (d_str, meal_id)).fetchall()
        else:
            rows = c.execute("SELECT name FROM menu_items WHERE menu_date=%s AND meal_id=%s ORDER BY id", (d_str, meal_id)).fetchall()
        if rows:
            return [r["name"] for r in rows]
    
    day_name = target_date.strftime("%A")
    return get_official_menu(day_name, meal_id, dietary)

@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return {
        "id": user["id"],
        "name": user["name"],
        "role": user["role"],
        "identifier": user["identifier"],
        "register_no": user["register_no"],
        "department": user["department"],
        "semester": user["semester"],
        "theme": user["theme"],
        "dietary_preference": user["dietary_preference"]
    }

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        verify_firebase_token(authorization.split(" ", 1)[1])
    return {"ok": True}

@app.put("/api/profile/theme")
def set_theme(body: ThemeBody, user=Depends(current_user)):
    if body.theme not in ("light", "dark"):
        raise HTTPException(400, "Theme must be light or dark")
    with connect() as c:
        c.execute("UPDATE users SET theme=%s WHERE id=%s", (body.theme, user["id"]))
        c.commit()
    return {"theme": body.theme}

@app.get("/api/menu/today")
def today_menu(
    dietary: str = Query(default="non_veg"),
    menu_date: Optional[str] = Query(default=None),
    user=Depends(current_user),
):
    now = datetime.now(TZ)
    current_date = now.date()
    selected_date = current_date
    if menu_date:
        try:
            selected_date = date.fromisoformat(menu_date)
        except ValueError:
            raise HTTPException(400, "Use YYYY-MM-DD for the menu date")
    is_today = selected_date == current_date
    active = active_meal(now) if is_today else None
    diet = "veg" if dietary.lower() == "veg" else "non_veg"
    selected_time = now.strftime("%H:%M") if is_today else ""
    
    meals = [
        {
            **m,
            "items": resolve_menu_items(selected_date, m["id"], diet),
            "status": "open" if active and active["id"] == m["id"] else ("upcoming" if is_today and selected_time < m["start"] else "closed")
        }
        for m in MEALS
    ]
    return {
        "date": selected_date.isoformat(),
        "date_label": selected_date.strftime("%a, %d %b %Y"),
        "active_meal": active["id"] if active else None,
        "dietary": diet,
        "meals": meals
    }

@app.get("/api/menu/complete")
def complete_menu(dietary: str = Query(default="non_veg"), user=Depends(current_user)):
    today = datetime.now(TZ).date()
    diet = "veg" if dietary.lower() == "veg" else "non_veg"
    days = []
    for i in range(7):
        target = today + timedelta(days=i)
        days.append({
            "date": target.isoformat(),
            "label": target.strftime("%A, %d %b"),
            "meals": [
                {
                    **m,
                    "items": resolve_menu_items(target, m["id"], diet)
                }
                for m in MEALS
            ]
        })
    return {"dietary": diet, "days": days}

@app.get("/api/feedback/mine")
def my_feedback(user=Depends(require_role("student"))):
    with connect() as c:
        ratings = [dict(r) for r in c.execute(
            "SELECT id,meal_id,service_date,rating,comment,category,created_at FROM feedback WHERE user_id=%s ORDER BY created_at DESC",
            (user["id"],)
        ).fetchall()]

        complaints = [dict(r) for r in c.execute(
            """SELECT c.id,c.meal_id,c.service_date,c.category,c.title,c.description,c.status,c.created_at,
                      g.response,g.status as group_status
               FROM complaints c
               LEFT JOIN complaint_group_links l ON l.complaint_id=c.id
               LEFT JOIN issue_groups g ON g.id=l.group_id
               WHERE c.user_id=%s ORDER BY c.created_at DESC""",
            (user["id"],)
        ).fetchall()]

    return {"ratings": ratings, "complaints": complaints}

@app.get("/api/feedback/responses")
def student_feedback_responses(user=Depends(require_role("student"))):
    """Return anonymized overall feedback summaries and manager responses by meal."""
    with connect() as c:
        rating_rows = c.execute(
            "SELECT meal_id, AVG(rating) AS average, COUNT(*) AS count FROM feedback GROUP BY meal_id"
        ).fetchall()
        issue_rows = c.execute(
            """SELECT meal_id, service_date, title, summary, count, status, response, responded_at
               FROM issue_groups
               ORDER BY service_date DESC, responded_at DESC, id DESC"""
        ).fetchall()

    ratings_by_meal = {
        row["meal_id"]: {"average": round(float(row["average"]), 1) if row["average"] is not None else None, "count": row["count"]}
        for row in rating_rows
    }
    issues_by_meal = {meal["id"]: [] for meal in MEALS}
    for row in issue_rows:
        if row["meal_id"] not in issues_by_meal:
            continue
        issues_by_meal[row["meal_id"]].append({
            "service_date": row["service_date"],
            "title": row["title"],
            "summary": row["summary"],
            "report_count": row["count"],
            "status": row["status"],
            "response": row["response"] or "",
            "responded_at": row["responded_at"],
        })

    return {
        "meals": [
            {
                "id": meal["id"],
                "name": meal["name"],
                "rating_average": ratings_by_meal.get(meal["id"], {}).get("average"),
                "rating_count": ratings_by_meal.get(meal["id"], {}).get("count", 0),
                "feedback_count": sum(issue["report_count"] for issue in issues_by_meal[meal["id"]]),
                "feedback_summaries": issues_by_meal[meal["id"]],
            }
            for meal in MEALS
        ]
    }

@app.post("/api/feedback")
def submit_feedback(body: FeedbackBody, user=Depends(require_role("student"))):
    meal = meal_by_id(body.meal_id)
    now = datetime.now(TZ)
    service_date = now.date().isoformat()

    active = active_meal(now)
    if not ALLOW_FEEDBACK_OUTSIDE_WINDOW and (not active or active["id"] != body.meal_id):
        raise HTTPException(403, f"{meal['name']} feedback is open {meal['time']} only.")

    now_iso = now.isoformat()
    with connect() as c:
        if not ALLOW_MULTIPLE_FEEDBACK:
            existing = c.execute(
                "SELECT 1 FROM feedback WHERE user_id=%s AND meal_id=%s AND service_date=%s LIMIT 1",
                (user["id"], body.meal_id, service_date),
            ).fetchone()
            if existing:
                raise HTTPException(400, "You have already submitted feedback for this meal today.")
        try:
            c.execute(
                """INSERT INTO feedback(user_id,meal_id,service_date,rating,items_json,comment,category,dietary,created_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (user["id"], body.meal_id, service_date, body.rating, json.dumps(body.items or {}), body.comment.strip()[:1000], body.category.strip()[:60], body.dietary, now_iso)
            )
            c.commit()
        except IntegrityError:
            raise HTTPException(400, "Feedback could not be saved. Check your rating and try again.")

    return {"ok": True, "message": "Thanks! Your feedback has been submitted."}

@app.post("/api/complaints")
def submit_complaint(body: ComplaintBody, background_tasks: BackgroundTasks, user=Depends(require_role("student"))):
    meal = meal_by_id(body.meal_id)
    now = datetime.now(TZ)
    service_date = now.date().isoformat()

    active = active_meal(now)
    if not active or active["id"] != body.meal_id:
        raise HTTPException(403, f"{meal['name']} feedback is open {meal['time']} only.")

    title = body.title.strip()
    desc = body.description.strip()
    if len(title) < 3:
        raise HTTPException(422, "Please provide a clearer title for your concern.")
    if len(desc) < 5:
        raise HTTPException(422, "Please share a few more details so the kitchen team can act on it.")

    now_iso = now.isoformat()
    with connect() as c:
        cur = c.execute(
            """INSERT INTO complaints(user_id,meal_id,service_date,category,title,description,status,processed_for_ai,created_at)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (user["id"], body.meal_id, service_date, body.category.strip()[:60], title[:120], desc[:2000], "Submitted", 0, now_iso)
        )
        c.commit()
        comp_id = cur.fetchone()["id"]

    # Queue Gemini AI processing in the background - zero wait for the student
    background_tasks.add_task(process_pending_complaints_job, connect, {m["id"]: m for m in MEALS}, TZ)

    return {"ok": True, "id": comp_id, "message": "Your complaint has been submitted privately."}

# ==========================================
# MANAGER & REVIEWER WORKSPACE ENDPOINTS
# ==========================================

@app.get("/api/manager/overview")
def manager_overview(user=Depends(require_roles("manager", "reviewer"))):
    reconcile_all_issues(connect)
    today = datetime.now(TZ).date().isoformat()
    with connect() as c:
        ratings = c.execute(
            "SELECT meal_id, AVG(rating) avg, COUNT(*) count FROM feedback WHERE service_date=%s GROUP BY meal_id",
            (today,)
        ).fetchall()
        total_complaints = c.execute("SELECT COUNT(*) n FROM complaints WHERE service_date=%s", (today,)).fetchone()["n"]
        rating_total = c.execute("SELECT COUNT(*) n FROM feedback WHERE service_date=%s", (today,)).fetchone()["n"]
        pending_ai = c.execute("SELECT COUNT(*) n FROM complaints WHERE processed_for_ai=0").fetchone()["n"]
        
        # Risk score summaries
        risk_counts = c.execute(
            """SELECT 
                 SUM(CASE WHEN risk_level='LOW' THEN 1 ELSE 0 END) as low_count,
                 SUM(CASE WHEN risk_level='MEDIUM' THEN 1 ELSE 0 END) as med_count,
                 SUM(CASE WHEN risk_level='HIGH' THEN 1 ELSE 0 END) as high_count,
                 SUM(CASE WHEN risk_level='CRITICAL' THEN 1 ELSE 0 END) as crit_count
               FROM issue_groups"""
        ).fetchone()
        active_escalations = c.execute("SELECT COUNT(*) as n FROM escalations WHERE status='Active'").fetchone()["n"]

        issues = c.execute(
            "SELECT * FROM issue_groups ORDER BY risk_score DESC, CASE status WHEN 'Open' THEN 0 ELSE 1 END, count DESC LIMIT 100"
        ).fetchall()

    meal_ratings = {r["meal_id"]: {"average": round(float(r["avg"]), 1) if r["avg"] is not None else 0.0, "count": r["count"]} for r in ratings}
    
    out = []
    now_dt = datetime.now(TZ)
    for row in issues:
        d = dict(row)
        d["overdue"] = d["status"] not in ("Responded", "Resolved") and now_dt > datetime.fromisoformat(d["deadline"])
        d["score_breakdown"] = {
            "recurrence": d.get("recurrence_score", 0),
            "delay": d.get("delay_score", 0),
            "credits": -(d.get("resolution_credits", 0)),
            "total": d.get("risk_score", 0)
        }
        out.append(d)

    ai_info = get_ai_status(pending_ai)

    return {
        "date": today,
        "total_complaints": total_complaints,
        "total_ratings": rating_total,
        "meal_ratings": meal_ratings,
        "risk_summary": {
            "low": risk_counts["low_count"] or 0,
            "medium": risk_counts["med_count"] or 0,
            "high": risk_counts["high_count"] or 0,
            "critical": risk_counts["crit_count"] or 0,
            "active_escalations": active_escalations
        },
        "issues": out,
        "ai_status": ai_info,
        "summary_mode": ai_info["provider"]
    }

@app.get("/api/manager/issues")
def get_manager_issues(
    meal_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    status: Optional[str] = None,
    escalated_only: bool = False,
    user=Depends(require_roles("manager", "reviewer"))
):
    reconcile_all_issues(connect)
    with connect() as c:
        query = "SELECT * FROM issue_groups WHERE 1=1"
        params = []
        if meal_id and meal_id != "all":
            query += " AND meal_id=%s"
            params.append(meal_id)
        if risk_level and risk_level != "all":
            query += " AND risk_level=%s"
            params.append(risk_level.upper())
        if status and status != "all":
            query += " AND status=%s"
            params.append(status)
        query += " ORDER BY risk_score DESC, count DESC"
        rows = c.execute(query, params).fetchall()

        issues_out = []
        now_dt = datetime.now(TZ)
        for r in rows:
            d = dict(r)
            esc = c.execute("SELECT * FROM escalations WHERE group_id=%s AND status='Active' LIMIT 1", (d["id"],)).fetchone()
            d["escalation"] = dict(esc) if esc else None
            cycle = c.execute("SELECT * FROM response_cycles WHERE group_id=%s ORDER BY cycle_number DESC LIMIT 1", (d["id"],)).fetchone()
            d["cycle"] = dict(cycle) if cycle else None
            d["overdue"] = d["status"] not in ("Responded", "Resolved") and now_dt > datetime.fromisoformat(d["deadline"])
            d["score_breakdown"] = {
                "recurrence": d.get("recurrence_score", 0),
                "delay": d.get("delay_score", 0),
                "credits": -(d.get("resolution_credits", 0)),
                "total": d.get("risk_score", 0)
            }
            if escalated_only and not d["escalation"]:
                continue
            issues_out.append(d)
    return {"issues": issues_out, "count": len(issues_out)}

@app.get("/api/manager/issues/{issue_id}")
def get_manager_issue_details(issue_id: int, user=Depends(require_roles("manager", "reviewer"))):
    with connect() as c:
        row = c.execute("SELECT * FROM issue_groups WHERE id=%s", (issue_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Issue not found")
        issue = dict(row)
        cycles = [dict(r) for r in c.execute("SELECT * FROM response_cycles WHERE group_id=%s ORDER BY cycle_number ASC", (issue_id,)).fetchall()]
        resolutions = [dict(r) for r in c.execute("SELECT * FROM resolutions WHERE group_id=%s ORDER BY id DESC", (issue_id,)).fetchall()]
        escalations = [dict(r) for r in c.execute("SELECT * FROM escalations WHERE group_id=%s ORDER BY id DESC", (issue_id,)).fetchall()]
        history = [dict(r) for r in c.execute("SELECT * FROM risk_history WHERE group_id=%s ORDER BY id DESC", (issue_id,)).fetchall()]
        # Privacy-guaranteed complaints (strictly NO user_id, student name, email, or reg_no)
        complaints = [
            {
                "id": r["id"],
                "meal_id": r["meal_id"],
                "service_date": r["service_date"],
                "category": r["category"],
                "title": r["title"],
                "description": r["description"],
                "status": r["status"],
                "created_at": r["created_at"]
            }
            for r in c.execute(
                """SELECT c.id, c.meal_id, c.service_date, c.category, c.title, c.description, c.status, c.created_at
                   FROM complaints c 
                   JOIN complaint_group_links l ON l.complaint_id=c.id 
                   WHERE l.group_id=%s""",
                (issue_id,)
            ).fetchall()
        ]
        issue["cycles"] = cycles
        issue["resolutions"] = resolutions
        issue["escalations"] = escalations
        issue["history"] = history
        issue["complaints"] = complaints
        issue["score_breakdown"] = {
            "recurrence": issue.get("recurrence_score", 0),
            "delay": issue.get("delay_score", 0),
            "credits": -(issue.get("resolution_credits", 0)),
            "total": issue.get("risk_score", 0)
        }
    return issue

@app.post("/api/manager/issues/{issue_id}/respond")
def respond_issue(issue_id: int, body: ResponseBody, user=Depends(require_roles("manager", "reviewer"))):
    response = body.response.strip()
    if len(response) < 3:
        raise HTTPException(422, "Please enter a valid response.")
    now = datetime.now(TZ).isoformat()
    with connect() as c:
        row = c.execute("SELECT id, deadline FROM issue_groups WHERE id=%s", (issue_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Issue not found")
        # Freeze response delay for active cycle
        c.execute(
            """UPDATE response_cycles 
               SET responded_at=%s, response_text=%s, status='Responded'
               WHERE group_id=%s AND status IN ('Open', 'Overdue')""",
            (now, response[:1500], issue_id)
        )
        c.execute(
            "UPDATE issue_groups SET response=%s, responded_at=%s, status='Responded' WHERE id=%s",
            (response[:1500], now, issue_id)
        )
        c.execute(
            "UPDATE complaints SET status='Responded' WHERE id IN (SELECT complaint_id FROM complaint_group_links WHERE group_id=%s)",
            (issue_id,)
        )
        # Recalculate score immediately
        reconcile_issue_risk(c, issue_id)
        c.commit()
    return {"ok": True, "responded_at": now, "message": "Manager response saved successfully."}

@app.post("/api/manager/issues/{issue_id}/verify-resolution")
def verify_resolution(issue_id: int, body: ResolutionBody, user=Depends(require_roles("reviewer", "manager"))):
    """
    Authorized resolution verification.
    Applies resolution credit (default 15 pts) and marks issue resolved.
    Enforces that duplicate credits cannot be granted for the same cycle/event.
    """
    credit = body.credit_amount if (body.credit_amount is not None and body.credit_amount > 0) else RESOLUTION_REDUCTION_POINTS
    now = datetime.now(TZ).isoformat()
    with connect() as c:
        issue = c.execute("SELECT * FROM issue_groups WHERE id=%s", (issue_id,)).fetchone()
        if not issue:
            raise HTTPException(404, "Issue not found")
        
        # Check if already verified in this cycle
        existing = c.execute(
            "SELECT id FROM resolutions WHERE group_id=%s AND trigger_type=%s",
            (issue_id, body.trigger_type)
        ).fetchone()
        if existing and issue["status"] == "Resolved":
            raise HTTPException(400, f"A {body.trigger_type} resolution has already been verified for this issue.")

        c.execute(
            """INSERT INTO resolutions(group_id, trigger_type, verified_by, credit_applied, notes, verified_at)
               VALUES(%s,%s,%s,%s,%s,%s)""",
            (issue_id, body.trigger_type, user.get("name") or user.get("identifier"), credit, body.notes.strip(), now)
        )
        c.execute(
            "UPDATE issue_groups SET status='Resolved' WHERE id=%s",
            (issue_id,)
        )
        c.execute(
            "UPDATE complaints SET status='Resolved' WHERE id IN (SELECT complaint_id FROM complaint_group_links WHERE group_id=%s)",
            (issue_id,)
        )
        reconcile_issue_risk(c, issue_id)
        c.commit()
        updated = c.execute("SELECT risk_score, risk_level, status FROM issue_groups WHERE id=%s", (issue_id,)).fetchone()
    return {
        "ok": True,
        "credit_applied": credit,
        "new_score": updated["risk_score"],
        "new_level": updated["risk_level"],
        "status": updated["status"],
        "message": f"Resolution verified. {credit} points credit applied."
    }

@app.get("/api/manager/escalations")
def get_escalations(status: Optional[str] = None, user=Depends(require_roles("manager", "reviewer"))):
    with connect() as c:
        query = """
            SELECT e.*, g.title as issue_title, g.meal_id, g.category, g.risk_score as current_score
            FROM escalations e
            JOIN issue_groups g ON g.id=e.group_id
            WHERE 1=1
        """
        params = []
        if status and status != "all":
            query += " AND e.status=%s"
            params.append(status)
        query += " ORDER BY CASE e.status WHEN 'Active' THEN 0 ELSE 1 END, e.triggered_at DESC"
        rows = [dict(r) for r in c.execute(query, params).fetchall()]
    return {"escalations": rows, "count": len(rows)}

@app.post("/api/manager/escalations/{escalation_id}/close")
def close_escalation(escalation_id: int, body: EscalationCloseBody, user=Depends(require_roles("reviewer", "manager"))):
    notes = body.closure_notes.strip()
    if len(notes) < 5:
        raise HTTPException(422, "Please provide comprehensive closure / inspection notes.")
    now = datetime.now(TZ).isoformat()
    with connect() as c:
        esc = c.execute("SELECT * FROM escalations WHERE id=%s", (escalation_id,)).fetchone()
        if not esc:
            raise HTTPException(404, "Escalation not found")
        if esc["status"] == "Closed":
            return {"ok": True, "message": "Escalation was already closed."}
        c.execute(
            """UPDATE escalations 
               SET status='Closed', closed_at=%s, closed_by=%s, closure_notes=%s
               WHERE id=%s""",
            (now, user.get("name") or user.get("identifier"), notes, escalation_id)
        )
        c.commit()
    return {"ok": True, "closed_at": now, "message": "Inspection escalation closed successfully."}

@app.post("/api/manager/issues/{issue_id}/override-safety")
def override_safety(issue_id: int, body: SafetyOverrideBody, user=Depends(require_roles("reviewer", "manager"))):
    with connect() as c:
        row = c.execute("SELECT id FROM issue_groups WHERE id=%s", (issue_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Issue not found")
        c.execute(
            "UPDATE issue_groups SET is_food_safety_hazard=%s WHERE id=%s",
            (1 if body.is_safety_hazard else 0, issue_id)
        )
        reconcile_issue_risk(c, issue_id)
        c.commit()
        updated = c.execute("SELECT risk_score, risk_level, is_food_safety_hazard FROM issue_groups WHERE id=%s", (issue_id,)).fetchone()
    return {"ok": True, "issue": dict(updated)}

@app.post("/api/manager/ai/trigger")
def trigger_ai_analysis(background_tasks: BackgroundTasks, user=Depends(require_roles("manager", "reviewer"))):
    background_tasks.add_task(process_pending_complaints_job, connect, {m["id"]: m for m in MEALS}, TZ)
    return {"ok": True, "message": "AI analysis queued in background."}

@app.get("/api/manager/ai/summary")
def manager_meal_ai_summary(meal_id: str, user=Depends(require_roles("manager", "reviewer"))):
    meal = meal_by_id(meal_id)
    today = datetime.now(TZ).date().isoformat()
    with connect() as c:
        complaints = [dict(row) for row in c.execute(
            """SELECT category,title,description,NULL AS rating FROM complaints
               WHERE meal_id=%s AND service_date=%s ORDER BY id""",
            (meal_id, today)
        ).fetchall()]
        ratings = [dict(row) for row in c.execute(
            """SELECT category, 'Meal rating' AS title, comment AS description, rating FROM feedback
               WHERE meal_id=%s AND service_date=%s ORDER BY id""",
            (meal_id, today)
        ).fetchall()]
    result = generate_meal_summary(meal["name"], complaints + ratings)
    return {**result, "meal_id": meal_id, "service_date": today, "count": len(complaints) + len(ratings)}

@app.get("/api/manager/feedback")
def manager_feedback(user=Depends(require_roles("manager", "reviewer"))):
    with connect() as c:
        groups = c.execute("SELECT * FROM issue_groups ORDER BY service_date DESC, count DESC").fetchall()
        ratings = c.execute("SELECT meal_id,service_date,AVG(rating) average,COUNT(*) count,category FROM feedback GROUP BY meal_id,service_date,category ORDER BY service_date DESC").fetchall()
        total_complaints = c.execute("SELECT COUNT(*) n FROM complaints").fetchone()["n"]
        total_ratings = c.execute("SELECT COUNT(*) n FROM feedback").fetchone()["n"]
        pending_ai = c.execute("SELECT COUNT(*) n FROM complaints WHERE processed_for_ai=0").fetchone()["n"]

    items = []
    for row in groups:
        d = dict(row)
        d.update({"type": "Complaint", "category": d.get("category") or "Grouped issue", "title": d["title"], "summary": d["summary"]})
        items.append(d)
    for row in ratings:
        d = dict(row)
        avg_val = round(float(d["average"]), 1) if d["average"] is not None else 0.0
        items.append({
            "id": f"rating-{d['meal_id']}-{d['service_date']}-{len(items)}",
            "type": "Rating",
            "meal_id": d["meal_id"],
            "service_date": d["service_date"],
            "category": d["category"] or "Meal rating",
            "title": f"Average rating {avg_val}/5",
            "summary": "Aggregated meal-rating responses for this meal and category; individual comments are not shown.",
            "rating": avg_val,
            "count": d["count"],
            "status": "Recorded"
        })

    return {
        "counts": {
            "all": len(items),
            "complaints": len(groups),
            "ratings": len(ratings),
            "total_complaints": total_complaints,
            "total_ratings": total_ratings,
            "pending_ai": pending_ai
        },
        "items": items,
        "ai_status": get_ai_status(pending_ai)
    }

@app.get("/api/manager/menu")
def manager_menu(menu_date: str, dietary: str = "non_veg", user=Depends(require_roles("manager", "reviewer"))):
    try:
        parsed_date = date.fromisoformat(menu_date)
    except ValueError:
        raise HTTPException(400, "Use YYYY-MM-DD for the menu date")
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM menu_items WHERE menu_date=%s ORDER BY CASE meal_id WHEN 'breakfast' THEN 1 WHEN 'lunch' THEN 2 WHEN 'snacks' THEN 3 ELSE 4 END, id",
            (menu_date,)
        ).fetchall()
        if not rows:
            day_name = parsed_date.strftime("%A")
            for meal in MEALS:
                items = get_official_menu(day_name, meal["id"], dietary)
                for item_name in items:
                    dtype = "Non-Veg" if any(w in item_name.lower() for w in ["chicken", "fish", "egg", "mutton", "omelet"]) else "Veg"
                    c.execute(
                        "INSERT INTO menu_items(menu_date,meal_id,name,description,dietary_type) VALUES(%s,%s,%s,%s,%s)",
                        (menu_date, meal["id"], item_name, "Official CEG Hostel Menu", dtype)
                    )
            c.commit()
            rows = c.execute(
                "SELECT * FROM menu_items WHERE menu_date=%s ORDER BY CASE meal_id WHEN 'breakfast' THEN 1 WHEN 'lunch' THEN 2 WHEN 'snacks' THEN 3 ELSE 4 END, id",
                (menu_date,)
            ).fetchall()
    return {"date": menu_date, "dietary": dietary, "items": [dict(r) for r in rows]}

@app.post("/api/manager/menu")
def save_manager_menu(body: MenuItemBody, user=Depends(require_roles("manager", "reviewer"))):
    try:
        date.fromisoformat(body.menu_date)
    except ValueError:
        raise HTTPException(400, "Use YYYY-MM-DD for the menu date")
    meal_by_id(body.meal_id)
    if body.dietary_type not in ("Veg", "Non-Veg", "Vegan"):
        raise HTTPException(400, "Invalid food type")
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Item name is required")
    with connect() as c:
        if body.id:
            exists = c.execute("SELECT id FROM menu_items WHERE id=%s", (body.id,)).fetchone()
            if not exists:
                raise HTTPException(404, "Menu item not found")
            c.execute(
                "UPDATE menu_items SET menu_date=%s,meal_id=%s,name=%s,description=%s,dietary_type=%s WHERE id=%s",
                (body.menu_date, body.meal_id, name, body.description[:250], body.dietary_type, body.id)
            )
            item_id = body.id
        else:
            cur = c.execute(
                "INSERT INTO menu_items(menu_date,meal_id,name,description,dietary_type) VALUES(%s,%s,%s,%s,%s) RETURNING id",
                (body.menu_date, body.meal_id, name, body.description[:250], body.dietary_type)
            )
            item_id = cur.fetchone()["id"]
        c.commit()
        row = c.execute("SELECT * FROM menu_items WHERE id=%s", (item_id,)).fetchone()
    return dict(row)

@app.delete("/api/manager/menu/{item_id}")
def delete_manager_menu(item_id: int, user=Depends(require_roles("manager", "reviewer"))):
    with connect() as c:
        cur = c.execute("DELETE FROM menu_items WHERE id=%s", (item_id,))
        c.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Menu item not found")
    return {"ok": True}
