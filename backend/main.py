import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent, WRITE_TOOLS
from memory import save_message, MEMORY_MODE
from tools import SF_MODE
from knowledge_base import QDRANT_MODE
from analytics import log_run, get_stats

# Registry of tools that /approve is allowed to execute.
# Approval is binding: we run EXACTLY the tool + args the user saw on the
# approval card — the agent is never re-run, so it can't change its mind.
WRITE_TOOL_REGISTRY = {t.name: t for t in WRITE_TOOLS}

# Server-side stash of the pending write per session. /chat records the exact
# tool + args the agent proposed; /approve executes THIS copy and ignores the
# args echoed by the client, so a tampered /approve body cannot change what
# runs. (Client-trusted args were the hole.)
# ponytail: in-memory dict, single pending write per session — fine for the
# single-worker free-tier deploy; move to the memory table if you scale out.
_PENDING: dict[str, dict] = {}

# SECTION 1: APP INSTANCE
app = FastAPI(
    title="Nexus360 API",
    version="3.1.0"
)

# SECTION 2: CORS
# Same-origin in the single-container deployment (no CORS needed there).
# ALLOWED_ORIGINS="https://a,https://b" adds extra origins for a separately
# hosted static frontend.
_extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"] + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SECTION 3: REQUEST / RESPONSE MODELS
class ChatRequest(BaseModel):
    message: str
    session_id: str

class ApprovalRequest(BaseModel):
    message: str
    session_id: str
    pending_tool: dict

class ChatResponse(BaseModel):
    output: str
    pending_tool: dict | None
    needs_approval: bool
    error: str | None


# SECTION 4: HEALTH ENDPOINTS
@app.get("/health")
def health():
    """Reports which mode each service is running in, so the UI can show
    an honest status instead of a hardcoded LIVE dot."""
    return {
        "status":     "ok",
        "salesforce": SF_MODE,      # "live" or "mock"
        "memory":     MEMORY_MODE,  # "supabase" or "sqlite"
        "qdrant":     QDRANT_MODE,  # "cloud" or "local"
    }

# Agent run analytics — tool mix, LLM-skip savings, approval rate, latency.
@app.get("/analytics")
def analytics():
    return get_stats()

# SECTION 5: CHAT ENDPOINT
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = run_agent(
        user_message=request.message,
        session_id=request.session_id,
    )

    needs_approval = result["pending_tool"] is not None

    # Stash the proposed write server-side so /approve runs exactly this,
    # not whatever the client posts back. A non-write turn supersedes and
    # clears any earlier pending write, so a stale card can't be approved.
    if needs_approval:
        _PENDING[request.session_id] = {
            "tool":    result["pending_tool"],
            "message": request.message,
        }
    else:
        _PENDING.pop(request.session_id, None)

    return ChatResponse(
        output = result["output"],
        pending_tool = result["pending_tool"],
        needs_approval = needs_approval,
        error = result["error"],
    )

# SECTION 6: APPROVE ENDPOINT
@app.post("/approve", response_model=ChatResponse)
def approve(request: ApprovalRequest):
    # Execute the write the SERVER stashed in /chat, never the client's args.
    pending = _PENDING.get(request.session_id)
    if pending is None:
        return ChatResponse(
            output="", pending_tool=None, needs_approval=False,
            error="No pending write for this session — nothing to approve.",
        )

    # Staleness gate: the card the user clicked must match the current pending
    # write. Multiple approval cards can sit in the transcript; approving an old
    # one must not run the newer stashed write. We still execute the SERVER copy
    # (never client args) — the echo is used only to detect a mismatch.
    echoed = request.pending_tool or {}
    if (echoed.get("name") != pending["tool"]["name"]
            or (echoed.get("args") or {}) != (pending["tool"].get("args") or {})):
        return ChatResponse(
            output="", pending_tool=None, needs_approval=False,
            error="This approval is stale — a newer request replaced it. Resend the request.",
        )

    name    = pending["tool"]["name"]
    args    = pending["tool"].get("args") or {}
    message = pending["message"]

    tool = WRITE_TOOL_REGISTRY.get(name)
    if tool is None:
        return ChatResponse(
            output="", pending_tool=None, needs_approval=False,
            error=f"'{name}' is not an approvable write operation.",
        )

    try:
        started = time.perf_counter()
        result  = tool.invoke(args)
        latency_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        return ChatResponse(
            output="", pending_tool=None, needs_approval=False, error=str(exc),
        )

    _PENDING.pop(request.session_id, None)
    log_run(request.session_id, [name], "write_approved", latency_ms)
    output = f"{str(result).strip()}\n\n(Source: Salesforce (write confirmed))"
    save_message(request.session_id, "user", message)
    save_message(request.session_id, "assistant", output)

    return ChatResponse(
        output=output, pending_tool=None, needs_approval=False, error=None,
    )


# SECTION 7: REJECT ENDPOINT
@app.post("/reject", response_model=ChatResponse)
def reject(request: ApprovalRequest):
    pending = _PENDING.pop(request.session_id, None)
    name    = pending["tool"]["name"] if pending else request.pending_tool.get("name", "the operation")
    message = pending["message"] if pending else request.message
    output  = f"Rejected — '{name}' was not executed. No changes were made."

    log_run(request.session_id, [name], "write_rejected", 0)
    # Save the turn so session memory matches what the user saw
    save_message(request.session_id, "user", message)
    save_message(request.session_id, "assistant", output)

    return ChatResponse(
        output=output, pending_tool=None, needs_approval=False, error=None,
    )

# SECTION 8: STATIC FRONTEND (production single-service deployment)
# When frontend/dist exists (built UI), serve it from FastAPI at / —
# one container, same origin, no CORS. In local dev (no dist), / returns
# API info and the UI runs separately on Vite :5173.
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")

if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="ui")
else:
    @app.get("/")
    def api_info():
        return {"status": "ok", "app": "Nexus360 API", "version": "3.1.0"}

# ENTRY POINT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
