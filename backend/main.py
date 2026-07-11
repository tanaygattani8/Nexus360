import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent, WRITE_TOOLS
from memory import save_message, MEMORY_MODE
from tools import SF_MODE
from knowledge_base import QDRANT_MODE

# Registry of tools that /approve is allowed to execute.
# Approval is binding: we run EXACTLY the tool + args the user saw on the
# approval card — the agent is never re-run, so it can't change its mind.
WRITE_TOOL_REGISTRY = {t.name: t for t in WRITE_TOOLS}

# SECTION 1: APP INSTANCE
app = FastAPI(
    title="Nexus360 API",
    version="3.1.0"
)

# SECTION 2: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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

# SECTION 5: CHAT ENDPOINT
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = run_agent(
        user_message=request.message,
        session_id=request.session_id,
    )

    needs_approval = result["pending_tool"] is not None

    return ChatResponse(
        output = result["output"],
        pending_tool = result["pending_tool"],
        needs_approval = needs_approval,
        error = result["error"],
    )

# SECTION 6: APPROVE ENDPOINT
@app.post("/approve", response_model=ChatResponse)
def approve(request: ApprovalRequest):
    name = request.pending_tool.get("name")
    args = request.pending_tool.get("args") or {}

    tool = WRITE_TOOL_REGISTRY.get(name)
    if tool is None:
        return ChatResponse(
            output="", pending_tool=None, needs_approval=False,
            error=f"'{name}' is not an approvable write operation.",
        )

    try:
        result = tool.invoke(args)
    except Exception as exc:
        return ChatResponse(
            output="", pending_tool=None, needs_approval=False, error=str(exc),
        )

    output = f"{str(result).strip()}\n\n(Source: Salesforce (write confirmed))"
    save_message(request.session_id, "user", request.message)
    save_message(request.session_id, "assistant", output)

    return ChatResponse(
        output=output, pending_tool=None, needs_approval=False, error=None,
    )


# SECTION 7: REJECT ENDPOINT
@app.post("/reject", response_model=ChatResponse)
def reject(request: ApprovalRequest):
    name = request.pending_tool.get("name", "the operation")
    output = f"Rejected — '{name}' was not executed. No changes were made."

    # Save the turn so session memory matches what the user saw
    save_message(request.session_id, "user", request.message)
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
