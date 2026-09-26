"""Vera challenge candidate bot.

Implements the 5-endpoint HTTP contract from challenge-testing-brief.md:
    POST /v1/context   GET /v1/healthz
    POST /v1/tick       GET /v1/metadata
    POST /v1/reply      POST /v1/teardown (optional, wipes state)

Run:
    pip install -r requirements.txt
    export LLM_PROVIDER=openai LLM_API_KEY=sk-...   # see .env.example
    uvicorn bot:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

import composer

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera-bot")

app = FastAPI(title="Vera Challenge Bot")
START = time.time()
STATE_LOCK = Lock()

TEAM_NAME = os.environ.get("TEAM_NAME", "Team Alpha")
TEAM_MEMBERS = [m.strip() for m in os.environ.get("TEAM_MEMBERS", "").split(",") if m.strip()]
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
BOT_VERSION = os.environ.get("BOT_VERSION", "1.0.0")
MAX_REPLY_TURNS = int(os.environ.get("MAX_REPLY_TURNS", "5"))
MAX_TICK_ACTIONS = 20

# ---------------------------------------------------------------------------
# In-memory state. Fine for the challenge's single test window; swap for
# Redis/SQLite if the bot needs to survive process restarts.
# ---------------------------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}          # (scope, context_id) -> {version, payload}
conversations: dict[str, dict] = {}                 # conversation_id -> conversation state
sent_suppression: dict[str, set[str]] = {}           # merchant_id -> {suppression_key, ...}
open_conversation: dict[tuple[str, Optional[str]], str] = {}  # (merchant_id, customer_id) -> conversation_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_payload(scope: str, context_id: Optional[str]) -> Optional[dict]:
    if not context_id:
        return None
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _reset_state() -> None:
    contexts.clear()
    conversations.clear()
    sent_suppression.clear()
    open_conversation.clear()


# ---------------------------------------------------------------------------
# /v1/context
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return {"accepted": False, "reason": "invalid_scope", "details": body.scope}

    key = (body.scope, body.context_id)
    with STATE_LOCK:
        current = contexts.get(key)
        if current and current["version"] >= body.version:
            return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}
        contexts[key] = {"version": body.version, "payload": body.payload}

    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": _now_iso()}


# ---------------------------------------------------------------------------
# /v1/healthz, /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts.keys():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": os.environ.get("LLM_MODEL") or f"{os.environ.get('LLM_PROVIDER', 'openai')}:default",
        "approach": "single-prompt composer over the 4-context framework, with regex-based "
                    "conversation heuristics (auto-reply detection, intent transition, "
                    "not-interested exit) steering the reply LLM call",
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": _now_iso(),
    }


@app.post("/v1/teardown")
def teardown():
    with STATE_LOCK:
        _reset_state()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /v1/tick
# ---------------------------------------------------------------------------

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
def tick(body: TickBody):
    actions: list[dict] = []

    for trigger_id in body.available_triggers:
        if len(actions) >= MAX_TICK_ACTIONS:
            break

        with STATE_LOCK:
            trigger = _get_payload("trigger", trigger_id)
            if not trigger:
                continue

            merchant_id = trigger.get("merchant_id")
            customer_id = trigger.get("customer_id")
            suppression_key = trigger.get("suppression_key", trigger_id)

            merchant = _get_payload("merchant", merchant_id)
            if not merchant:
                continue
            category = _get_payload("category", merchant.get("category_slug"))
            if not category:
                continue
            customer = _get_payload("customer", customer_id) if customer_id else None

            if suppression_key in sent_suppression.get(merchant_id, set()):
                continue  # already sent for this trigger/window

            lock_key = (merchant_id, customer_id)
            if lock_key in open_conversation:
                continue  # one active outbound thread per merchant(/customer) at a time

        try:
            composed = composer.compose_initial(category, merchant, trigger, customer)
        except Exception as exc:  # noqa: BLE001 - never crash a tick over one bad trigger
            log.warning("compose_initial failed for trigger=%s: %s", trigger_id, exc)
            continue

        conversation_id = f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:6]}"
        send_as = "merchant_on_behalf" if customer_id else "vera"

        action = {
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": [merchant.get("identity", {}).get("name", "")],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": suppression_key,
            "rationale": composed["rationale"],
        }

        with STATE_LOCK:
            sent_suppression.setdefault(merchant_id, set()).add(suppression_key)
            open_conversation[lock_key] = conversation_id
            conversations[conversation_id] = {
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "trigger_id": trigger_id,
                "send_as": send_as,
                "turns": [{"from": send_as, "body": composed["body"]}],
                "sent_bodies": {composer.normalize_body(composed["body"])},
                "ended": False,
            }

        actions.append(action)

    return {"actions": actions}


# ---------------------------------------------------------------------------
# /v1/reply
# ---------------------------------------------------------------------------

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
def reply(body: ReplyBody):
    with STATE_LOCK:
        convo = conversations.get(body.conversation_id)
        if convo is None:
            # Judge referenced a conversation we don't recognize (shouldn't happen in
            # practice since conversation_ids originate from our own /v1/tick actions).
            convo = {
                "merchant_id": body.merchant_id,
                "customer_id": body.customer_id,
                "trigger_id": None,
                "send_as": "merchant_on_behalf" if body.customer_id else "vera",
                "turns": [],
                "sent_bodies": set(),
                "ended": False,
            }
            conversations[body.conversation_id] = convo

        convo["turns"].append({"from": body.from_role, "body": body.message})
        merchant_id = convo["merchant_id"] or body.merchant_id
        customer_id = convo["customer_id"] or body.customer_id
        trigger_id = convo.get("trigger_id")

        merchant = _get_payload("merchant", merchant_id)
        category = _get_payload("category", merchant.get("category_slug")) if merchant else None
        trigger = _get_payload("trigger", trigger_id)
        customer = _get_payload("customer", customer_id) if customer_id else None

        hints = composer.detect_hints(convo["turns"], body.message, body.turn_number, MAX_REPLY_TURNS)
        turns_snapshot = list(convo["turns"])
        sent_bodies = convo["sent_bodies"]

    try:
        result = composer.compose_reply(category, merchant, trigger, customer, turns_snapshot, hints)
    except Exception as exc:  # noqa: BLE001 - degrade to a graceful end, don't crash the call
        log.warning("compose_reply failed for conversation=%s: %s", body.conversation_id, exc)
        result = {"action": "end", "rationale": f"composer_error: {exc}"}

    with STATE_LOCK:
        if result["action"] == "send":
            norm = composer.normalize_body(result["body"])
            if norm in sent_bodies:
                # Anti-repetition guard: never resend a verbatim body.
                result = {"action": "end", "rationale": "avoided sending a verbatim repeat"}
            else:
                sent_bodies.add(norm)
                turn = {"from": convo["send_as"], "body": result["body"]}
                if hints["auto_reply_suspected"]:
                    # Tag this as the one nudge we send on suspected auto-reply, so if the
                    # other side sends another canned reply, detect_hints escalates to
                    # auto_reply_confirmed instead of nudging forever.
                    turn["auto_reply_nudge"] = True
                convo["turns"].append(turn)

        if result["action"] == "end" or hints["turn_limit_reached"]:
            convo["ended"] = True
            lock_key = (merchant_id, customer_id)
            if open_conversation.get(lock_key) == body.conversation_id:
                del open_conversation[lock_key]

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
