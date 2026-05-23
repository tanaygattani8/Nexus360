# ── Imports ───────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# ── Supabase client ───────────────────────────────────────────────────────────
# Created once at module level — reused across all calls
def _get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL or SUPABASE_KEY not set in .env")
    return create_client(url, key)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MESSAGE
# ══════════════════════════════════════════════════════════════════════════════
# Called after every user message and every agent response.
# session_id groups all messages in one conversation together.
# role is either "user" or "assistant".

def save_message(session_id: str, role: str, content: str) -> None:
    """
    Save a single message to Supabase.

    Args:
        session_id: Unique ID for this conversation (e.g. a UUID or timestamp).
        role:       "user" or "assistant"
        content:    The message text.
    """
    try:
        client = _get_client()
        client.table("conversations").insert({
            "session_id": session_id,
            "role":       role,
            "content":    content,
        }).execute()
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
    Load the last N messages for a session from Supabase.

    Args:
        session_id: The session to load history for.
        limit:      How many past messages to include. Default 10.
                    Keep this low — every message uses LLM context window.

    Returns:
        List of dicts: [{"role": "user", "content": "..."}, ...]
        Ordered oldest to newest.
    """
    try:
        client = _get_client()
        result = (
            client.table("conversations")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)   # get latest first
            .limit(limit)
            .execute()
        )
        # Reverse so oldest message comes first (correct order for LLM)
        messages = list(reversed(result.data))
        return messages
    except Exception as e:
        print(f"⚠️  Memory load failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CLEAR HISTORY  (utility — useful for testing)
# ══════════════════════════════════════════════════════════════════════════════

def clear_history(session_id: str) -> None:
    """Delete all messages for a session. Useful for resetting during demos."""
    try:
        client = _get_client()
        client.table("conversations").delete().eq("session_id", session_id).execute()
        print(f"🗑️  Cleared history for session: {session_id}")
    except Exception as e:
        print(f"⚠️  Memory clear failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  python memory.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_SESSION = "test-session-001"

    print("── Saving messages ──────────────────────────────────")
    save_message(TEST_SESSION, "user",      "What is the health of Acme Corp?")
    save_message(TEST_SESSION, "assistant", "Acme Corp has 2 open high-priority cases.")
    save_message(TEST_SESSION, "user",      "What about their opportunities?")
    print("✅ 3 messages saved")

    print("\n── Loading history ──────────────────────────────────")
    history = load_history(TEST_SESSION)
    for msg in history:
        print(f"  [{msg['role']}] {msg['content']}")

    print("\n── Clearing history ─────────────────────────────────")
    clear_history(TEST_SESSION)

    print("\n── Confirming cleared ───────────────────────────────")
    history = load_history(TEST_SESSION)
    print(f"  Messages remaining: {len(history)}")
    print("✅ Done")
