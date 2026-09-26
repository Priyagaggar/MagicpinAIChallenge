"""Prompt building, response parsing, and conversation heuristics for the Vera composer.

Keeps all "what makes a good Vera message" knowledge (rubric, anti-patterns,
compulsion levers) in one place, per challenge-brief.md sections 5, 8, 10, 11.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm import chat

SYSTEM_PROMPT = """You are the message-composition engine for Vera, magicpin's AI assistant for \
merchant growth. Vera talks to merchants (and, on their behalf, to their customers) over WhatsApp.

You are given up to four context layers for a single message decision:
  1. CATEGORY  - slow-changing knowledge about this kind of business (voice, offer catalog, \
peer benchmarks, research digest, seasonal beats, trend signals).
  2. MERCHANT  - this specific business's current state (identity, performance, offers, \
conversation history, customer aggregates, derived signals).
  3. TRIGGER   - the event that justifies messaging right now.
  4. CUSTOMER  - (optional) the merchant's own customer, only present for customer-facing sends.

Hard rules:
- Never fabricate a fact, number, offer, citation, or name that isn't present in the given \
contexts. If something isn't in the contexts, don't say it.
- Every number you cite must support the SAME claim it supports in the source context. Never \
detach a real statistic from its original subject and reattach it to a different claim (e.g. a \
digest item's "X% more calls from a GBP tag" cannot become "X% more calls from a festival \
offer" - that is fabrication even though X% is a real number).
- Only reach for a category's `digest` item when the trigger itself is digest/research-shaped \
(e.g. kind contains "digest", "research", "compliance", "trend") or the digest item is the most \
relevant fact available. For other trigger kinds (festival, perf_spike, recall_due, ...), ground \
the message in the trigger's own payload and the merchant's own offers/performance instead of \
importing an unrelated digest stat.
- Anchor the message on a concrete, verifiable fact from the contexts (a number, date, \
headline, or peer stat). Generic framings ("increase your sales", "flat discount") lose.
- When offering a bundle or combo, only combine items that are actually bundled together in the \
merchant's own `offers` or the category's `offer_catalog` as a single entry - don't invent a new \
pairing by stitching two separate catalog entries into one offer.
- Match the category voice exactly (e.g. dentists/pharmacies = clinical peer tone, no "cure" or \
"guaranteed"; salons/restaurants/gyms = warmer but still not hype-promotional).
- Prefer service+price offers already in the merchant's catalog ("Haircut @ ₹299") over \
generic discount language.
- Exactly one primary call-to-action. Binary yes/no for action triggers, open-ended for \
informational/curiosity triggers, or none for pure FYI.
- Put the actual ask in the last sentence. No long preambles, no re-introducing yourself.
- Honor the merchant's or customer's language preference; Hindi-English code-mix is encouraged \
when the language preference includes "hi".
- Never repeat a message you can see was already sent verbatim in this conversation.
- Use one or more of these compulsion levers: specificity/verifiability, loss aversion, social \
proof, effort externalization (offer to do the work), curiosity, reciprocity, asking the \
merchant a direct question, single binary commitment.
- send_as is "vera" for merchant-facing messages, "merchant_on_behalf" for customer-facing ones.

You will be scored on: specificity, category fit, merchant fit, trigger relevance, and \
engagement compulsion (0-10 each). You must respond with ONLY a single JSON object, no \
markdown fences, no commentary."""

INITIAL_INSTRUCTIONS = """Compose the next outbound WhatsApp message for this situation.

Respond with ONLY this JSON object:
{
  "body": "<the message text>",
  "cta": "binary" | "open_ended" | "none",
  "rationale": "<one sentence: why this message, why now, what it should achieve>"
}"""

REPLY_INSTRUCTIONS = """The other party just replied. Decide what to do next.

Behavioral hints you MUST follow if present and true:
- If hints.auto_reply_suspected is true and this is the first time: send ONE short message \
asking the merchant to check directly themselves (don't just repeat what you already said).
- If hints.auto_reply_confirmed is true: end the conversation gracefully and politely - don't \
send another nudge, you already tried once.
- If hints.not_interested is true: end the conversation gracefully, thank them, no hard sell.
- If hints.intent_go_ahead is true: do NOT ask another qualifying question. Move straight into \
executing or confirming the concrete next step.
- If hints.turn_limit_reached is true: end the conversation gracefully regardless of content.
- If the message is hostile or abusive, stay polite and do not escalate. If it also contains an \
unrelated question, give a brief honest answer (or say it's outside what you can help with) and \
gently redirect back to the original topic.

Respond with ONLY one of these JSON shapes:
{"action": "send", "body": "<text>", "cta": "binary" | "open_ended" | "none", "rationale": "<why>"}
{"action": "wait", "wait_seconds": <int>, "rationale": "<why>"}
{"action": "end", "rationale": "<why>"}"""


def _trim(obj: Any, max_list: int = 8) -> Any:
    """Trim long lists inside a context payload so prompts stay small and fast."""
    if isinstance(obj, dict):
        return {k: _trim(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim(v, max_list) for v in obj[:max_list]]
    return obj


def _context_block(category: Optional[dict], merchant: Optional[dict],
                    trigger: Optional[dict], customer: Optional[dict]) -> str:
    parts = {
        "category": _trim(category),
        "merchant": _trim(merchant),
        "trigger": _trim(trigger),
        "customer": _trim(customer),
    }
    return json.dumps(parts, ensure_ascii=False, indent=2)


def build_initial_prompt(category: dict, merchant: dict, trigger: dict,
                          customer: Optional[dict]) -> str:
    return (
        f"{_context_block(category, merchant, trigger, customer)}\n\n{INITIAL_INSTRUCTIONS}"
    )


def build_reply_prompt(category: Optional[dict], merchant: Optional[dict], trigger: Optional[dict],
                        customer: Optional[dict], conversation_turns: list[dict],
                        hints: dict) -> str:
    convo = json.dumps(conversation_turns, ensure_ascii=False, indent=2)
    hints_json = json.dumps(hints, ensure_ascii=False)
    return (
        f"{_context_block(category, merchant, trigger, customer)}\n\n"
        f"Conversation so far (oldest first):\n{convo}\n\n"
        f"hints = {hints_json}\n\n{REPLY_INSTRUCTIONS}"
    )


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip common markdown code-fence wrapping.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


VALID_CTA = {"binary", "open_ended", "none"}


def compose_initial(category: dict, merchant: dict, trigger: dict,
                     customer: Optional[dict]) -> dict:
    """Returns {body, cta, rationale}. Raises LLMError/ValueError on failure."""
    prompt = build_initial_prompt(category, merchant, trigger, customer)
    raw = chat(SYSTEM_PROMPT, prompt, temperature=0.0)
    parsed = _extract_json(raw)
    body = str(parsed.get("body", "")).strip()
    cta = str(parsed.get("cta", "none")).strip()
    if cta not in VALID_CTA:
        cta = "open_ended"
    if not body:
        raise ValueError("LLM returned an empty body")
    return {
        "body": body,
        "cta": cta,
        "rationale": str(parsed.get("rationale", "")).strip() or "Composed from category+merchant+trigger context.",
    }


def compose_reply(category: Optional[dict], merchant: Optional[dict], trigger: Optional[dict],
                   customer: Optional[dict], conversation_turns: list[dict], hints: dict) -> dict:
    """Returns {action, body?, cta?, wait_seconds?, rationale}."""
    prompt = build_reply_prompt(category, merchant, trigger, customer, conversation_turns, hints)
    raw = chat(SYSTEM_PROMPT, prompt, temperature=0.0)
    parsed = _extract_json(raw)
    action = str(parsed.get("action", "")).strip().lower()
    if action not in {"send", "wait", "end"}:
        action = "end"
    out: dict = {"action": action, "rationale": str(parsed.get("rationale", "")).strip() or "—"}
    if action == "send":
        body = str(parsed.get("body", "")).strip()
        if not body:
            raise ValueError("LLM returned action=send with an empty body")
        cta = str(parsed.get("cta", "none")).strip()
        out["body"] = body
        out["cta"] = cta if cta in VALID_CTA else "open_ended"
    elif action == "wait":
        try:
            out["wait_seconds"] = max(1, int(parsed.get("wait_seconds", 900)))
        except (TypeError, ValueError):
            out["wait_seconds"] = 900
    return out


# ---------------------------------------------------------------------------
# Conversation heuristics (auto-reply detection, intent transition, etc.)
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    r"thank you for (contacting|your message|reaching out)",
    r"we (will|shall) (get back|revert|respond) to you",
    r"automated (assistant|reply|response|message)",
    r"this is an auto([- ]?generated| ?reply)",
    r"hamari team tak pahuncha",
    r"aap ki jaankari.{0,20}shukriya",
    r"currently (unavailable|out of office|closed)",
    r"business hours",
    r"i am (a bot|an automated)",
]
_AUTO_REPLY_RE = re.compile("|".join(AUTO_REPLY_PATTERNS), re.IGNORECASE)

NOT_INTERESTED_PATTERNS = [
    r"not interested", r"\bstop\b", r"no thanks", r"unsubscribe",
    r"\bband karo\b", r"mat bhejo", r"don'?t (message|contact) me",
]
_NOT_INTERESTED_RE = re.compile("|".join(NOT_INTERESTED_PATTERNS), re.IGNORECASE)

INTENT_GO_PATTERNS = [
    r"\byes\b.{0,15}(let'?s|go|do it|please)", r"go ahead", r"let'?s do it",
    r"\bsure\b", r"\bok(ay)?\b.{0,10}(go|do it|please)?", r"\bhaan\b", r"\bchalo\b",
    r"please proceed", r"start (it|now)",
]
_INTENT_GO_RE = re.compile("|".join(INTENT_GO_PATTERNS), re.IGNORECASE)

PROFANITY_RE = re.compile(r"\b(idiot|stupid|scam|fraud|useless|bloody)\b", re.IGNORECASE)


def normalize_body(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_hints(conversation_turns: list[dict], latest_message: str, turn_number: int,
                  max_turns: int = 5) -> dict:
    """Cheap regex/repetition heuristics that steer the LLM's reply decision."""
    incoming = [t for t in conversation_turns if t.get("from") in ("merchant", "customer")]
    norm_latest = normalize_body(latest_message)
    prior_same = sum(1 for t in incoming[:-1] if normalize_body(t.get("body", "")) == norm_latest)

    auto_reply_pattern_hit = bool(_AUTO_REPLY_RE.search(latest_message))
    auto_reply_repeat_hit = prior_same >= 1
    already_nudged = any(t.get("auto_reply_nudge") for t in conversation_turns)

    auto_reply_suspected = (auto_reply_pattern_hit or auto_reply_repeat_hit) and not already_nudged
    auto_reply_confirmed = (auto_reply_pattern_hit or auto_reply_repeat_hit) and already_nudged

    return {
        "auto_reply_suspected": auto_reply_suspected,
        "auto_reply_confirmed": auto_reply_confirmed,
        "not_interested": bool(_NOT_INTERESTED_RE.search(latest_message)),
        "intent_go_ahead": bool(_INTENT_GO_RE.search(latest_message)),
        "hostile": bool(PROFANITY_RE.search(latest_message)),
        "turn_limit_reached": turn_number >= max_turns,
    }
