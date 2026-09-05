# ── Agent run analytics ───────────────────────────────────────────────────────
# Lightweight telemetry for every agent turn: which tools fired, which response
# path was taken (and so whether an LLM synthesis call was skipped), latency,
# and write approval/rejection. This is the "measurement layer" — it recreates,
# in miniature, the metrics Salesforce ships as Agentforce Command Center
# (tool/action mix, latency, cost per conversation, approval rate).
#
# Storage is ALWAYS local SQLite, independent of the Supabase/SQLite memory
# mode: it is demo telemetry, not user data, so it needs no cloud table and
# works out of the box with only GROQ_API_KEY set. Ephemeral on Render (resets
# on restart), same as the SQLite chat memory — fine for a demo.

import os
import sqlite3

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.db")

# Response paths, cheapest to most expensive. "template" and "direct" skip the
# LLM synthesis call; "llm" pays for it. Writes get their own approval outcomes.
PATHS = ("direct", "template", "llm", "approval_pending", "write_approved", "write_rejected")

# ponytail: $0.10 per LLM synthesis call is a stated demo assumption for the
# "cost saved" headline, loosely tracking Salesforce Flex Credits (~$0.10/action).
# Not a billing figure — swap it if you have a real per-call number.
_COST_PER_LLM_CALL = 0.10


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            tools      TEXT NOT NULL,   -- comma-joined tool names, "" if none
            path       TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


def log_run(session_id: str, tools: list[str], path: str, latency_ms: int) -> None:
    """Record one agent turn. Non-fatal: telemetry must never break a chat."""
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO runs (session_id, tools, path, latency_ms) VALUES (?, ?, ?, ?)",
                (session_id, ",".join(tools), path, int(latency_ms)),
            )
    except Exception as e:
        print(f"[analytics] log failed: {e}")


def clear_runs(session_id: str) -> None:
    """Delete a session's telemetry. Used by the eval suite so evaluation
    traffic doesn't inflate the dashboard's real numbers."""
    try:
        with _conn() as conn:
            conn.execute("DELETE FROM runs WHERE session_id = ?", (session_id,))
    except Exception as e:
        print(f"[analytics] clear failed: {e}")


def _percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile. Empty → 0."""
    if not values:
        return 0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered)) - 1))
    return ordered[k]


def get_stats() -> dict:
    """Aggregate all runs into the numbers the dashboard renders."""
    try:
        with _conn() as conn:
            rows = conn.execute("SELECT tools, path, latency_ms FROM runs").fetchall()
    except Exception as e:
        print(f"[analytics] read failed: {e}")
        rows = []

    total = len(rows)
    by_path = {p: 0 for p in PATHS}
    tool_counts: dict[str, int] = {}
    latencies: list[int] = []

    for tools, path, latency_ms in rows:
        by_path[path] = by_path.get(path, 0) + 1
        latencies.append(latency_ms)
        for t in (tools.split(",") if tools else []):
            tool_counts[t] = tool_counts.get(t, 0) + 1

    # LLM-skip savings: only "template" turns skipped a synthesis call they
    # otherwise would have made. "direct" (no-tool) turns never had a tool
    # result to synthesize, so they don't count as an avoided call.
    skipped = by_path.get("template", 0)
    llm_calls = by_path.get("llm", 0)
    tool_turns = skipped + llm_calls
    skip_rate = round(skipped / tool_turns * 100) if tool_turns else 0

    # Approval rate over resolved writes only.
    approved = by_path.get("write_approved", 0)
    rejected = by_path.get("write_rejected", 0)
    resolved = approved + rejected
    approval_rate = round(approved / resolved * 100) if resolved else 0

    return {
        "total_runs":       total,
        "by_path":          by_path,
        "tool_counts":      tool_counts,
        "llm_calls_skipped": skipped,
        "skip_rate":        skip_rate,          # % of tool turns that skipped LLM synthesis
        "est_cost_saved":   round(skipped * _COST_PER_LLM_CALL, 2),
        "approval_rate":    approval_rate,      # % of resolved writes approved
        "writes_approved":  approved,
        "writes_rejected":  rejected,
        "latency_p50_ms":   _percentile(latencies, 50),
        "latency_p95_ms":   _percentile(latencies, 95),
    }


# SMOKE TEST  —  python analytics.py
if __name__ == "__main__":
    import tempfile
    # isolate: run against a throwaway db so we don't pollute real telemetry
    _DB_PATH = os.path.join(tempfile.gettempdir(), "nexus360_analytics_test.db")
    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)

    log_run("s1", ["get_account_health"], "template", 120)
    log_run("s1", ["search_knowledge_base"], "llm", 900)
    log_run("s1", [], "direct", 40)
    log_run("s1", ["update_opportunity_stage"], "approval_pending", 200)
    log_run("s1", ["update_opportunity_stage"], "write_approved", 300)
    log_run("s2", ["create_support_case"], "write_rejected", 150)

    s = get_stats()
    print(s)
    assert s["total_runs"] == 6
    assert s["by_path"]["template"] == 1 and s["by_path"]["llm"] == 1
    # skipped = template(1) only; tool_turns = template(1) + llm(1) = 2 → 50%
    assert s["llm_calls_skipped"] == 1
    assert s["skip_rate"] == 50
    assert s["est_cost_saved"] == 0.10
    # 1 approved of 2 resolved → 50%
    assert s["approval_rate"] == 50
    assert s["tool_counts"]["update_opportunity_stage"] == 2
    assert s["latency_p50_ms"] > 0 and s["latency_p95_ms"] >= s["latency_p50_ms"]
    print("OK — analytics aggregation verified")
    try:  # best-effort: Windows keeps the sqlite file handle until GC
        os.remove(_DB_PATH)
    except OSError:
        pass
