# Vera Challenge — Submission

## Approach

A FastAPI service implementing the 5-endpoint contract (`/v1/context`, `/v1/tick`, `/v1/reply`,
`/v1/healthz`, `/v1/metadata`). Two composition paths, both a single LLM call with
temperature=0 for determinism:

- **`/v1/tick`** — for each hinted trigger, resolves its (category, merchant, trigger, customer?)
  contexts and calls `composer.compose_initial`. One system prompt encodes the full rubric
  (specificity, category fit, merchant fit, trigger relevance, engagement compulsion), the
  anti-pattern list, and the compulsion levers from `challenge-brief.md`. Dedup is by
  `suppression_key` per merchant, and only one open conversation per `(merchant, customer)` is
  allowed at a time — the bot won't fire a second thread at a merchant it's already talking to.

- **`/v1/reply`** — cheap regex heuristics (`composer.detect_hints`) classify the incoming
  message first: auto-reply (canned phrases + verbatim-repeat detection), hard "not interested",
  intent go-ahead ("yes let's do it" style), hostile/abusive, and turn-limit. These are passed to
  the LLM as explicit hints it must obey (e.g. "don't ask another qualifying question if
  `intent_go_ahead`"), rather than trusting the LLM to notice them unprompted — this made the
  auto-reply-hell and intent-transition replay scenarios much more reliable than pure
  LLM-judgment. An anti-repetition guard rejects any body identical to one already sent in that
  conversation and ends the conversation instead.

## Tradeoffs

- **In-memory state**, not Redis/SQLite — fine for a single 60-minute test window, not for
  surviving a restart. Traded persistence for zero infra to stand up.
- **Regex heuristics over an LLM classifier** for auto-reply/intent/not-interested detection —
  faster, free, deterministic, and the patterns are narrow enough (drawn directly from the
  brief's example conversations) that false positives are rare. A learned classifier would
  generalize better to auto-reply phrasing we haven't seen.
- **One open conversation per merchant at a time** — avoids spamming a merchant with two
  triggers' worth of messages in parallel, at the cost of possibly delaying a lower-priority
  trigger until the current thread ends. Didn't rank triggers by `urgency` to pick which one
  gets skipped; first-seen wins.
- **Single-provider-per-deployment LLM client** (`llm.py`) via raw HTTP, no SDKs — keeps the
  dependency footprint small and makes provider-swapping a one-line env change, at the cost of
  not using each provider's native retry/streaming niceties.

## What additional context would have helped most

- **Trigger `urgency` isn't used for tick-time prioritization** — with 20 actions/tick and
  multiple triggers active for the same merchant, knowing which one the judge considers most
  time-sensitive (beyond just "urgency: 1-5") would sharpen the pick.
- **No signal for "has this merchant been messaged today by any means"** outside this bot's own
  state — a cross-channel fatigue signal would help avoid over-messaging in a real deployment.
