"""
Gemini AI Analysis Service for MessMate
Uses Google's `google-genai` SDK with fast Gemini Flash models.
Ensures zero latency impact on student actions by running strictly in background tasks.
Provides robust fallback to deterministic rule-based analysis if API is absent or unavailable.
Integrates directly with the MessMate Risk-Score Engine.
Fully migrated to PostgreSQL / Neon.
"""

import os
import json
import re
import threading
from collections import Counter
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, ".env"))

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from app.risk_engine import (
    get_calendar_week_key,
    reconcile_issue_risk,
    calculate_risk_score
)

# Thread-safe lock to prevent duplicate concurrent AI analysis runs
_analysis_lock = threading.Lock()
_is_analyzing = False

def get_gemini_config():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip()
    return api_key, model

def is_ai_enabled() -> bool:
    api_key, _ = get_gemini_config()
    return bool(api_key and genai is not None)

def get_ai_status(pending_count: int = 0) -> Dict[str, Any]:
    api_key, model = get_gemini_config()
    return {
        "enabled": bool(api_key and genai is not None),
        "is_analyzing": _is_analyzing,
        "pending_count": pending_count,
        "model": model if api_key else "deterministic-fallback",
        "provider": "google-genai" if (api_key and genai is not None) else "rule-based",
    }

def normalise_rule_based(text: str):
    t = text.lower()
    is_safety = any(w in t for w in ["poison", "cockroach", "glass", "insect", "worm", "bug", "rotten", "vomit", "hospital", "dead", "contamination"])
    if any(w in t for w in ["cold", "not warm", "temperature", "chilled", "lukewarm"]):
        return "food_temperature", "Food temperature", "Food was served cold", "Several students reported that the meal was not served at a suitable temperature.", 0.85, is_safety
    if any(w in t for w in ["salt", "salty"]):
        return "too_salty", "Food taste", "Food was too salty", "Feedback indicates that seasoning or salt level was excessively strong.", 0.85, is_safety
    if any(w in t for w in ["hard", "dry", "softness", "chewy", "burnt"]):
        return "texture", "Food quality", "Food texture needs attention", "Students reported concerns about the texture or softness of the food.", 0.80, is_safety
    if any(w in t for w in ["hygiene", "clean", "hair", "unclean", "fly", "insect", "cockroach", "spoon", "plate"]):
        return "hygiene", "Hygiene and cleanliness", "Hygiene and cleanliness concern", "A hygiene concern was submitted regarding cleanliness of utensils, food, or dining hall.", 0.90, is_safety
    if any(w in t for w in ["quantity", "portion", "less food", "ran out", "insufficient", "empty"]):
        return "portion_size", "Food quantity", "Insufficient food quantity", "Feedback suggests that serving quantity was insufficient or food ran out.", 0.85, is_safety
    if any(w in t for w in ["veg", "non-veg", "chicken in veg", "meat in veg"]):
        return "veg_mismatch", "Vegetarian/non-vegetarian menu mismatch", "Vegetarian / Non-Vegetarian mismatch", "A concern was raised regarding separation or labeling of vegetarian and non-vegetarian food.", 0.90, is_safety
    if any(w in t for w in ["late", "wait", "queue", "slow", "delay"]):
        return "service_time", "Service and waiting time", "Service and waiting time issue", "Students experienced delays or long lines during meal service.", 0.85, is_safety
    slug = re.sub(r"[^a-z0-9]+", "_", t).strip("_")[:48] or "general_feedback"
    title = (text.strip().capitalize()[:70] or "General food quality feedback")
    return slug, "Food quality", title, "A recurring student concern was reported. Review the submitted feedback pattern and decide on an appropriate action.", 0.75, is_safety

def call_gemini_grouping(complaints: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Calls Google Gemini Flash to group complaints into issues.
    Never sends student identities, emails, or personal data.
    """
    api_key, model_name = get_gemini_config()
    if not api_key or genai is None:
        return None

    client = genai.Client(api_key=api_key)
    
    # Strip any potential PII - send only id, meal_id, category, title, description
    payload_data = [
        {
            "id": c["id"],
            "meal_id": c["meal_id"],
            "category": c.get("category", "General"),
            "title": c.get("title", ""),
            "description": c.get("description", "")
        }
        for c in complaints
    ]

    prompt = (
        "You are an expert college mess feedback intelligence analyst. "
        "Analyze the following student complaints for a campus dining hall and group them into actionable issue themes.\n"
        "Strict rules:\n"
        "1. Do not invent facts, complaints, or counts.\n"
        "2. Only assign valid complaint IDs from the input list to complaint_ids.\n"
        "3. Keep student identities anonymous; focus purely on the food/service problem.\n"
        "4. Assign a normalized category from: ['Food quality', 'Food taste', 'Food temperature', 'Food quantity', 'Hygiene and cleanliness', 'Menu mismatch', 'Service and waiting time', 'Vegetarian/non-vegetarian menu mismatch', 'Other', 'Unclear'].\n"
        "5. Set is_food_safety_hazard: true ONLY IF an urgent food safety hazard is reported (e.g. food poisoning, foreign objects like glass/metal/stones, insects/pests, spoiled/decayed food).\n"
        "6. Return a JSON array of grouped issues matching this schema:\n"
        "[\n"
        "  {\n"
        "    \"issue_key\": \"short_snake_case_key\",\n"
        "    \"category\": \"Food temperature\",\n"
        "    \"title\": \"Clear concise issue title (max 70 chars)\",\n"
        "    \"summary\": \"One to two sentence summary of the aggregated problem (max 200 chars)\",\n"
        "    \"confidence\": 0.95,\n"
        "    \"is_food_safety_hazard\": false,\n"
        "    \"complaint_ids\": [1, 2]\n"
        "  }\n"
        "]\n\n"
        f"Input complaints:\n{json.dumps(payload_data)}"
    )

    models_to_try = [model_name] + [m for m in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"] if m != model_name]
    for candidate in models_to_try:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1000,
                ),
            )
            text = response.text or ""
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list) and len(parsed) > 0:
                    print(f"[GeminiService] Successfully clustered {len(complaints)} complaints using {candidate}")
                    return parsed
        except Exception as e:
            print(f"[GeminiService] AI grouping attempt with {candidate} failed: {e}")
            continue
    return None

def generate_meal_summary(meal_name: str, feedback: List[Dict[str, Any]]) -> Dict[str, str]:
    """Summarize one meal's anonymized feedback, with a data-based fallback."""
    if not feedback:
        return {"summary": f"No feedback has been recorded for {meal_name} today.", "provider": "rule-based"}

    entries = [
        {
            "category": str(item.get("category") or "General"),
            "title": str(item.get("title") or ""),
            "feedback": str(item.get("description") or item.get("comment") or ""),
            "rating": item.get("rating"),
        }
        for item in feedback
    ]
    categories = Counter(entry["category"] for entry in entries)
    common_category, category_count = categories.most_common(1)[0]
    examples = list(dict.fromkeys(
        entry["feedback"] or entry["title"] for entry in entries if entry["feedback"] or entry["title"]
    ))[:3]
    fallback = (
        f"{len(entries)} feedback item(s) were recorded for {meal_name} today. "
        f"The most common theme is {common_category.lower()} ({category_count} report(s)). "
        f"Students most often noted: {'; '.join(examples)}"
    )

    api_key, model_name = get_gemini_config()
    if not api_key or genai is None:
        return {"summary": fallback, "provider": "rule-based"}

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=(
                "Summarize the following anonymized campus meal feedback in 2 concise sentences. "
                "Accurately report the dominant concern and count. Do not invent facts, "
                "identify students, or include recommendations not supported by the feedback. "
                f"Meal: {meal_name}. Total feedback items: {len(entries)}.\n"
                f"Feedback: {json.dumps(entries[:100], ensure_ascii=False)}"
            ),
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=220),
        )
        summary = (response.text or "").strip()
        if summary:
            return {"summary": summary, "provider": "google-genai"}
    except Exception as e:
        print(f"[GeminiService] Meal summary generation failed: {e}")

    return {"summary": fallback, "provider": "rule-based"}

def process_pending_complaints_job(connect_fn, meal_info_dict, tz):
    """
    Background job triggered on new complaint submission or on demand.
    Ensures non-blocking execution, idempotent operations, and deterministic fallback.
    Updates risk scores, weekly recurrence records, and response cycles.
    """
    global _is_analyzing
    if not _analysis_lock.acquire(blocking=False):
        return

    _is_analyzing = True
    try:
        with connect_fn() as c:
            unprocessed = c.execute(
                "SELECT id, meal_id, service_date, category, title, description, created_at FROM complaints WHERE processed_for_ai=0 ORDER BY id ASC LIMIT 50"
            ).fetchall()
            
            if not unprocessed:
                return

            complaints_list = [dict(row) for row in unprocessed]
            
            by_period = {}
            for comp in complaints_list:
                key = (comp["meal_id"], comp["service_date"])
                by_period.setdefault(key, []).append(comp)

            now_iso = datetime.now(tz).isoformat()

            for (meal_id, service_date), comp_batch in by_period.items():
                gemini_result = call_gemini_grouping(comp_batch) if is_ai_enabled() else None
                
                if gemini_result:
                    for g in gemini_result:
                        key = str(g.get("issue_key", "general_issue"))[:48]
                        cat = str(g.get("category", "Food quality"))[:50]
                        title = str(g.get("title", "Food Feedback"))[:100]
                        summary = str(g.get("summary", ""))[:300]
                        conf = float(g.get("confidence", 0.9))
                        is_hazard = 1 if g.get("is_food_safety_hazard") else 0

                        # Filter strictly to valid complaint IDs from this batch
                        c_ids = [cid for cid in g.get("complaint_ids", []) if any(x["id"] == cid for x in comp_batch)]
                        if not c_ids:
                            continue
                        
                        existing = c.execute(
                            "SELECT id, count, status FROM issue_groups WHERE meal_id=%s AND issue_key=%s",
                            (meal_id, key)
                        ).fetchone()
                        
                        m_end = meal_info_dict.get(meal_id, {}).get("end", "22:00")
                        deadline = datetime.combine(date.fromisoformat(service_date), datetime.strptime(m_end, "%H:%M").time(), tz) + timedelta(hours=24)
                        deadline_iso = deadline.isoformat()

                        try:
                            s_dt = datetime.fromisoformat(service_date)
                            w_key = get_calendar_week_key(s_dt)
                        except Exception:
                            w_key = get_calendar_week_key(datetime.now(tz))

                        if existing:
                            gid = existing["id"]
                            c.execute(
                                """UPDATE issue_groups 
                                   SET count=count+%s, title=%s, summary=%s, category=%s, confidence=%s, 
                                       is_food_safety_hazard=GREATEST(is_food_safety_hazard, %s),
                                       last_reported_at=%s
                                   WHERE id=%s""",
                                (len(c_ids), title, summary, cat, conf, is_hazard, now_iso, gid)
                            )
                            # If previously resolved/responded, open a new response cycle
                            if existing["status"] in ("Responded", "Resolved"):
                                max_cycle = c.execute("SELECT COALESCE(MAX(cycle_number), 1) as m FROM response_cycles WHERE group_id=%s", (gid,)).fetchone()["m"]
                                c.execute(
                                    """INSERT INTO response_cycles(group_id, cycle_number, started_at, deadline, status)
                                       VALUES(%s,%s,%s,%s, 'Open')
                                       ON CONFLICT (group_id, cycle_number) DO NOTHING""",
                                    (gid, max_cycle + 1, now_iso, deadline_iso)
                                )
                                c.execute("UPDATE issue_groups SET status='Reopened', deadline=%s WHERE id=%s", (deadline_iso, gid))
                        else:
                            cur = c.execute(
                                """INSERT INTO issue_groups(
                                       meal_id, service_date, issue_key, category, title, summary, count, 
                                       confidence, is_food_safety_hazard, first_reported_at, last_reported_at, deadline, status
                                   ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, 'Open') RETURNING id""",
                                (meal_id, service_date, key, cat, title, summary, len(c_ids), conf, is_hazard, now_iso, now_iso, deadline_iso)
                            )
                            gid = cur.fetchone()["id"]
                            c.execute(
                                """INSERT INTO response_cycles(group_id, cycle_number, started_at, deadline, status)
                                   VALUES(%s, 1, %s, %s, 'Open')
                                   ON CONFLICT (group_id, cycle_number) DO NOTHING""",
                                (gid, now_iso, deadline_iso)
                            )
                        
                        # Record week recurrence record (deduplicated!)
                        c.execute(
                            """INSERT INTO issue_week_records(group_id, week_key, first_seen_at)
                               VALUES(%s,%s,%s)
                               ON CONFLICT (group_id, week_key) DO NOTHING""",
                            (gid, w_key, now_iso)
                        )

                        for cid in c_ids:
                            c.execute(
                                """INSERT INTO complaint_group_links(complaint_id, group_id)
                                   VALUES(%s,%s)
                                   ON CONFLICT (complaint_id) DO UPDATE SET group_id = EXCLUDED.group_id""",
                                (cid, gid)
                            )
                            c.execute("UPDATE complaints SET processed_for_ai=1 WHERE id=%s", (cid,))
                        
                        # Reconcile risk score for this issue group immediately
                        reconcile_issue_risk(c, gid)
                else:
                    # Deterministic rule-based fallback
                    for comp in comp_batch:
                        key, cat, title, summary, conf, is_hazard = normalise_rule_based(comp["category"] + " " + comp["title"] + " " + comp["description"])
                        existing = c.execute(
                            "SELECT id, status FROM issue_groups WHERE meal_id=%s AND issue_key=%s",
                            (meal_id, key)
                        ).fetchone()
                        m_end = meal_info_dict.get(meal_id, {}).get("end", "22:00")
                        deadline = datetime.combine(date.fromisoformat(service_date), datetime.strptime(m_end, "%H:%M").time(), tz) + timedelta(hours=24)
                        deadline_iso = deadline.isoformat()

                        try:
                            s_dt = datetime.fromisoformat(service_date)
                            w_key = get_calendar_week_key(s_dt)
                        except Exception:
                            w_key = get_calendar_week_key(datetime.now(tz))

                        is_hazard_int = 1 if is_hazard else 0

                        if existing:
                            gid = existing["id"]
                            c.execute(
                                """UPDATE issue_groups 
                                   SET count=count+1, category=%s, confidence=%s, 
                                       is_food_safety_hazard=GREATEST(is_food_safety_hazard, %s),
                                       last_reported_at=%s
                                   WHERE id=%s""",
                                (cat, conf, is_hazard_int, now_iso, gid)
                            )
                            if existing["status"] in ("Responded", "Resolved"):
                                max_cycle = c.execute("SELECT COALESCE(MAX(cycle_number), 1) as m FROM response_cycles WHERE group_id=%s", (gid,)).fetchone()["m"]
                                c.execute(
                                    """INSERT INTO response_cycles(group_id, cycle_number, started_at, deadline, status)
                                       VALUES(%s,%s,%s,%s, 'Open')
                                       ON CONFLICT (group_id, cycle_number) DO NOTHING""",
                                    (gid, max_cycle + 1, now_iso, deadline_iso)
                                )
                                c.execute("UPDATE issue_groups SET status='Reopened', deadline=%s WHERE id=%s", (deadline_iso, gid))
                        else:
                            cur = c.execute(
                                """INSERT INTO issue_groups(
                                       meal_id, service_date, issue_key, category, title, summary, count, 
                                       confidence, is_food_safety_hazard, first_reported_at, last_reported_at, deadline, status
                                   ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, 'Open') RETURNING id""",
                                (meal_id, service_date, key, cat, title, summary, 1, conf, is_hazard_int, now_iso, now_iso, deadline_iso)
                            )
                            gid = cur.fetchone()["id"]
                            c.execute(
                                """INSERT INTO response_cycles(group_id, cycle_number, started_at, deadline, status)
                                   VALUES(%s, 1, %s, %s, 'Open')
                                   ON CONFLICT (group_id, cycle_number) DO NOTHING""",
                                (gid, now_iso, deadline_iso)
                            )

                        c.execute(
                            """INSERT INTO issue_week_records(group_id, week_key, first_seen_at)
                               VALUES(%s,%s,%s)
                               ON CONFLICT (group_id, week_key) DO NOTHING""",
                            (gid, w_key, now_iso)
                        )
                        c.execute(
                            """INSERT INTO complaint_group_links(complaint_id, group_id)
                               VALUES(%s,%s)
                               ON CONFLICT (complaint_id) DO UPDATE SET group_id = EXCLUDED.group_id""",
                            (comp["id"], gid)
                        )
                        c.execute("UPDATE complaints SET processed_for_ai=1 WHERE id=%s", (comp["id"],))
                        reconcile_issue_risk(c, gid)

            # Record AI processing job
            c.execute(
                """INSERT INTO ai_processing_jobs(batch_size, status, model, provider, created_at)
                   VALUES(%s,%s,%s,%s,%s)""",
                (len(complaints_list), "completed", get_gemini_config()[1], "google-genai" if is_ai_enabled() else "rule-based", now_iso)
            )
            c.commit()
    except Exception as e:
        print(f"[GeminiService] Error during background job: {e}")
    finally:
        _is_analyzing = False
        _analysis_lock.release()
