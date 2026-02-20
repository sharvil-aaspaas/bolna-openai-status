"""
status_fetch_slack.py
─────────────────────────────────────────────────────────────────────────────
Event-based OpenAI API status monitor via Slack Events API.

Source: https://status.openai.com  (checked 2026-02-19)
The "APIs" group on the OpenAI status page contains 12 components:
  Chat Completions, Responses, Fine-tuning, Embeddings, Images,
  Batch, Audio, Moderations, Realtime, Files, Login, Sora

Only events that mention one of these components are printed.
Non-API events (e.g. ChatGPT, Sora standalone) are silently dropped.

Slack message format received from openAI Statuspage:

  ✅ OpenAI - Incident resolved
  Sora 2 Degraded Performance

  Status: Resolved

  All impacted services have now fully recovered.

Pipeline:
  OpenAI Status Page → Slack channel (#openai-status) → Slack Events API
      → This Flask server → filtered, structured console output

Setup checklist (one-time):
  1. Create Slack workspace + #openai-status channel.
  2. Subscribe channel to OpenAI status updates:
       https://status.openai.com → "Subscribe to Updates" → Slack tab
  3. Create Slack App: https://api.slack.com/apps
       • Enable Event Subscriptions
       • Request URL: https://<your-ngrok>.ngrok.app/slack/events
       • Subscribe to bot event: message.channels
       • Install app to workspace, add bot to #openai-status
  4. Set env var: export SLACK_SIGNING_SECRET=<from api.slack.com>
  5. Run:  python status_fetch_slack.py
  6. Run:  ngrok http 5000   (separate terminal)
"""

import hashlib
import hmac
import logging
import os
import re
import time
from collections import deque
from datetime import datetime

from flask import Flask, abort, request, jsonify

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

SLACK_SIGNING_SECRET: str | None = os.getenv("SLACK_SIGNING_SECRET")
PORT: int = int(os.getenv("PORT", 5000))

# ── In-memory event log (last 100 events) ─────────────────────────────────────
# Stored as dicts so they can be rendered in the public /logs page.
EVENT_LOG: deque = deque(maxlen=100)

# ── API component filter ───────────────────────────────────────────────────────
# Exact component names from https://status.openai.com → "APIs" group (12 total).
# Only Slack messages that reference at least one of these will be logged.
API_COMPONENTS = {
    "chat completions",
    "responses",
    "fine-tuning",
    "fine tuning",
    "embeddings",
    "images",
    "batch",
    "audio",
    "moderations",
    "realtime",
    "files",
    "login",
    "sora",
}

# Regex built from the component names so we can do a fast, case-insensitive search.
_API_COMPONENT_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in API_COMPONENTS) + r")\b",
    re.IGNORECASE,
)

# Atlassian Statuspage status keywords.
_STATUS_RE = re.compile(
    r"\b(Investigating|Identified|Monitoring|Resolved|In Progress|"
    r"Operational|Degraded Performance|Partial Outage|Major Outage|"
    r"Under Maintenance|Postmortem|Update)\b",
    re.IGNORECASE,
)

# Emoji → human-readable severity map (Atlassian Statuspage conventions).
# Both unicode (typed manually) and Slack emoji codes (sent by RSS bot) are mapped.
_EMOJI_SEVERITY = {
    "✅": "Resolved",
    ":white_check_mark:": "Resolved",
    "🟡": "Degraded / Minor",
    ":large_yellow_circle:": "Degraded / Minor",
    "🟠": "Partial Outage",
    ":large_orange_circle:": "Partial Outage",
    "🔴": "Major Outage",
    ":red_circle:": "Major Outage",
    "🔵": "Maintenance",
    ":large_blue_circle:": "Maintenance",
    "ℹ️": "Informational",
    ":information_source:": "Informational",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def verify_slack_signature(req) -> bool:
    """
    Verify the request originated from Slack using HMAC-SHA256.
    https://docs.slack.dev/authentication/verifying-requests-from-slack

    If SLACK_SIGNING_SECRET is not set, verification is skipped with a warning
    (acceptable during local dev; MUST be set in production).
    """
    if not SLACK_SIGNING_SECRET:
        log.warning(
            "SLACK_SIGNING_SECRET not set — skipping signature verification. "
            "Set it in production!"
        )
        return True

    timestamp: str = req.headers.get("X-Slack-Request-Timestamp", "")
    slack_sig: str = req.headers.get("X-Slack-Signature", "")

    try:
        if abs(time.time() - float(timestamp)) > 300:
            log.warning("Rejected stale request (replay attack guard).")
            return False
    except (ValueError, TypeError):
        log.warning("Rejected: invalid timestamp header.")
        return False

    base = f"v0:{timestamp}:{req.get_data(as_text=True)}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), base.encode(), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, slack_sig):
        log.warning("Rejected: Slack signature mismatch.")
        return False

    return True


def is_api_event(text: str) -> bool:
    """
    Return True only if the message body references at least one of the
    12 API components listed on https://status.openai.com.
    """
    return bool(_API_COMPONENT_RE.search(text))


def parse_statuspage_message(text: str) -> dict:
    """
    Parse the plain-text Slack message posted by Atlassian Statuspage.

    Expected format (from real Slack payload):
        ✅ OpenAI - Incident resolved
        Sora 2 Degraded Performance

        Status: Resolved

        All impacted services have now fully recovered.

    Returns:
        {
            "headline":   str,   # first line, e.g. "✅ OpenAI - Incident resolved"
            "component":  str,   # matched API component name
            "status":     str,   # e.g. "Resolved"
            "body":       str,   # update detail text
            "severity":   str,   # derived from leading emoji
        }
    """
    lines = [ln.strip() for ln in text.strip().splitlines()]
    non_empty = [ln for ln in lines if ln]

    result = {
        "headline": non_empty[0] if non_empty else text,
        "component": "",
        "status": "",
        "body": "",
        "severity": "",
    }

    # Detect severity from leading emoji on the first line.
    for emoji, label in _EMOJI_SEVERITY.items():
        if result["headline"].startswith(emoji):
            result["severity"] = label
            break

    # Extract "Status: <value>" line.
    for line in non_empty:
        status_match = re.match(r"^Status\s*:\s*(.+)$", line, re.IGNORECASE)
        if status_match:
            result["status"] = status_match.group(1).strip()
            break

    # Fallback: extract status keyword from full text.
    if not result["status"]:
        kw = _STATUS_RE.search(text)
        if kw:
            result["status"] = kw.group(0).title()

    # Extract matched API component name.
    comp_match = _API_COMPONENT_RE.search(text)
    if comp_match:
        result["component"] = comp_match.group(0).title()

    # Body: everything after the "Status: ..." line.
    body_lines = []
    past_status = False
    for line in non_empty:
        if re.match(r"^Status\s*:", line, re.IGNORECASE):
            past_status = True
            continue
        if past_status and line:
            body_lines.append(line)
    result["body"] = " ".join(body_lines).strip()

    return result


def log_api_event(parsed: dict) -> None:
    """Print a formatted API status event to the console and store it."""
    sep = "=" * 55
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(sep)
    print(f"  🔔  OpenAI API Status Event — {ts}")
    print(f"  Component : {parsed['component'] or 'Unknown'}")
    print(f"  Status    : {parsed['status'] or 'Unknown'}")
    if parsed["severity"]:
        print(f"  Severity  : {parsed['severity']}")
    print(f"  Headline  : {parsed['headline']}")
    if parsed["body"]:
        print(f"  Details   : {parsed['body']}")
    print(sep)

    # Store event for the public /logs page.
    EVENT_LOG.appendleft({
        "timestamp": ts,
        "component": parsed["component"] or "Unknown",
        "status": parsed["status"] or "Unknown",
        "severity": parsed["severity"] or "Unknown",
        "headline": parsed["headline"],
        "body": parsed["body"],
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok", "service": "openai-api-status-monitor"}), 200


@app.route("/logs", methods=["GET"])
def public_logs():
    """
    Public read-only view of received API status events.
    Auto-refreshes every 30 s. No authentication required.
    """
    severity_colors = {
        "Resolved": "#22c55e",
        "Degraded / Minor": "#eab308",
        "Partial Outage": "#f97316",
        "Major Outage": "#ef4444",
        "Maintenance": "#3b82f6",
        "Informational": "#6b7280",
        "Unknown": "#6b7280",
    }

    rows = ""
    for e in EVENT_LOG:
        color = severity_colors.get(e["severity"], "#6b7280")
        rows += f"""
        <tr>
          <td>{e['timestamp']}</td>
          <td>{e['component']}</td>
          <td><span style="color:{color};font-weight:600">{e['status']}</span></td>
          <td><span style="color:{color};font-weight:600">{e['severity']}</span></td>
          <td>{e['headline']}</td>
          <td>{e['body'] or '—'}</td>
        </tr>"""

    if not rows:
        rows = '<tr><td colspan="6" style="text-align:center;color:#888">No events received yet.</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="30">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAI API Status Monitor — Live Logs</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
    h1   {{ font-size: 1.4rem; margin-bottom: 4px; }}
    p    {{ color: #94a3b8; font-size: 0.85rem; margin: 0 0 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th   {{ background: #1e293b; padding: 10px 14px; text-align: left;
            color: #94a3b8; font-weight: 600; border-bottom: 1px solid #334155; }}
    td   {{ padding: 10px 14px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
    tr:hover td {{ background: #1e293b; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:999px;
              background:#1e293b; font-size:0.75rem; }}
  </style>
</head>
<body>
  <h1>🔔 OpenAI API Status Monitor — Live Logs</h1>
  <p>Showing last {len(EVENT_LOG)} event(s). Page auto-refreshes every 30 s.
     Only API-group events are shown (Chat Completions, Embeddings, Audio, etc.)</p>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th><th>Component</th><th>Status</th>
        <th>Severity</th><th>Headline</th><th>Details</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Receives all Slack Events API payloads.

    Flow:
      1. Verify Slack signature (security).
      2. If url_verification → echo challenge (one-time handshake).
      3. If event_callback → extract message text → filter to API events only
         → parse & log.
      4. Always return 200 OK within 3 s (Slack retries on anything else).
    """
    if not verify_slack_signature(request):
        abort(403)

    data: dict = request.json or {}
    event_type: str = data.get("type", "")

    # ── 1. URL Verification (Slack handshake) ─────────────────────────────────
    if event_type == "url_verification":
        log.info("Slack URL verification handshake — responding with challenge.")
        return jsonify({"challenge": data.get("challenge", "")}), 200

    # ── 2. Event callback ─────────────────────────────────────────────────────
    if event_type == "event_callback":
        event: dict = data.get("event", {})

        # Process fresh channel messages.
        # Allow "bot_message" (RSS feed from Slackbot) but drop edits/deletions.
        IGNORED_SUBTYPES = {"message_changed", "message_deleted", "message_replied"}
        subtype = event.get("subtype", "")
        if event.get("type") == "message" and subtype not in IGNORED_SUBTYPES:

            # Statuspage posts arrive as plain text inside the message.
            # Rich attachments may also be present; check both.
            text: str = event.get("text", "")

            # Supplement with attachment fallback if plain text is thin.
            attachments = event.get("attachments", [])
            if not text and attachments:
                text = (
                    attachments[0].get("fallback")
                    or attachments[0].get("text")
                    or attachments[0].get("title")
                    or ""
                )

            if not text:
                log.debug("Empty message received — ignoring.")
                return jsonify({"status": "ok"}), 200

            # ── API component filter ──────────────────────────────────────────
            if not is_api_event(text):
                log.info(
                    "Non-API event dropped: %.80s…",
                    text.replace("\n", " "),
                )
                return jsonify({"status": "ok"}), 200

            # ── Parse & log ───────────────────────────────────────────────────
            parsed = parse_statuspage_message(text)
            log_api_event(parsed)

    # Slack requires 200 OK in < 3 s.
    return jsonify({"status": "ok"}), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("OpenAI API Status Monitor — Slack Events Listener")
    log.info("Endpoint : http://0.0.0.0:%d/slack/events", PORT)
    log.info("Filter   : APIs group (12 components from status.openai.com)")
    log.info(
        "Signing  : %s",
        "ENABLED ✅" if SLACK_SIGNING_SECRET else "DISABLED ⚠️  (set SLACK_SIGNING_SECRET)"
    )
    log.info("=" * 55)
    app.run(host="0.0.0.0", port=PORT, debug=False)
