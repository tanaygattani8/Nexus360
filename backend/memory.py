# ── Imports ───────────────────────────────────────────────────────────────────
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# ── Storage backend ───────────────────────────────────────────────────────────
# Supabase when configured, otherwise a local SQLite file with the same schema.
# The SQLite fallback means the demo keeps working when the free Supabase
# project pauses or expires — just remove SUPABASE_URL from .env.

MEMORY_MODE = "supabase" if (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY")) else "sqlite"
_SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")

_supabase = None  # created once, reused across calls

if MEMORY_MODE == "sqlite":
    print(f"[memory] No Supabase credentials — using local SQLite memory at {_SQLITE_PATH}")


def _get_client():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _supabase


def _sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MESSAGE
# ══════════════════════════════════════════════════════════════════════════════
# Called after every user message and every agent response.
# session_id groups all messages in one conversation together.
# role is either "user" or "assistant".

def save_message(session_id: str, role: str, content: str) -> None:
    """
    Save a single message.

    Args:
        session_id: Unique ID for this conversation (e.g. a UUID or timestamp).
        role:       "user" or "assistant"
        content:    The message text.
    """
    try:
        if MEMORY_MODE == "supabase":
            _get_client().table("conversations").insert({
                "session_id": session_id,
                "role":       role,
                "content":    content,
            }).execute()
        else:
            with _sqlite() as conn:
                conn.execute(
                    "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content),
                )
    except Exception as e:
        # Non-fatal — if memory save fails, the agent still works
        print(f"⚠️  Memory save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD HISTORY
# ══════════════════════════════════════════════════════════════════════════════
# Called at the start of every agent run to inject past context into the LLM.
# Returns the last N messages ordered oldest → newest so the LLM reads them
# in the right sequence.

def load_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Load the last N messages for a session.

    Args:
        session_id: The session to load history for.
        limit:      How many past messages to include. Default 10.
                    Keep this low — every message uses LLM context window.

    Returns:
        List of dicts: [{"role": "user", "content": "..."}, ...]
        Ordered oldest to newest.
    """
    try:
        if MEMORY_MODE == "supabase":
            result = (
                _get_client().table("conversations")
                .select("role, content, created_at")
                .eq("session_id", session_id)
                .order("created_at", desc=True)   # get latest first
                .limit(limit)
                .execute()
            )
            # Reverse so oldest message comes first (correct order for LLM)
            return list(reversed(result.data))
        else:
            with _sqlite() as conn:
                rows = conn.execute(
                    "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            return [{"role": r, "content": c} for r, c in reversed(rows)]
    except Exception as e:
        print(f"⚠️  Memory load failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CLEAR HISTORY  (utility — useful for testing)
# ══════════════════════════════════════════════════════════════════════════════

def clear_history(session_id: str) -> None:
    """Delete all messages for a session. Useful for resetting during demos."""
    try:
        if MEMORY_MODE == "supabase":
            _get_client().table("conversations").delete().eq("session_id", session_id).execute()
        else:
            with _sqlite() as conn:
                conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        print(f"🗑️  Cleared history for session: {session_id}")
    except Exception as e:
        print(f"⚠️  Memory clear failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  python memory.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_SESSION = "test-session-001"
    print(f"Memory mode: {MEMORY_MODE}")

    print("── Saving messages ──────────────────────────────────")
    save_message(TEST_SESSION, "user",      "What is the health of Acme Corp?")
    save_message(TEST_SESSION, "assistant", "Acme Corp has 2 open high-priority cases.")
    save_message(TEST_SESSION, "user",      "What about their opportunities?")
    print("✅ 3 messages saved")

    print("\n── Loading history ──────────────────────────────────")
    history = load_history(TEST_SESSION)
    for msg in history:
        print(f"  [{msg['role']}] {msg['content']}")
    assert len(history) == 3, f"expected 3 messages, got {len(history)}"
    assert history[0]["role"] == "user" and history[-1]["role"] == "user"

    print("\n── Clearing history ─────────────────────────────────")
    clear_history(TEST_SESSION)

    print("\n── Confirming cleared ───────────────────────────────")
    history = load_history(TEST_SESSION)
    print(f"  Messages remaining: {len(history)}")
    assert len(history) == 0
    print("✅ Done")
