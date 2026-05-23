from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import run_agent

# SECTION 1: APP INSTANCE
app = FastAPI(
    title="Nexus360 API",
    version="3.0.0"
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


# SECTION 4: HEALTH ENDPOINT
@app.get("/")
def health_check():
    return {"status": "ok", "app": "Nexus360 API", "version": "3.0.0"}

# SECTION 5: CHAT ENDPOINT
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = run_agent(
        user_message=request.message,
        session_id=request.session_id,
    )

    needs_approval = (
        result["pending_tool"] is not None and
        result["approved"] is None
    )

    return ChatResponse(
        output = result["output"],
        pending_tool = result["pending_tool"],
        needs_approval = needs_approval,
        error = result["error"],
    )

# SECTION 6: APPROVE ENDPOINT
@app.post("/approve", response_model=ChatResponse)
def approve(request: ApprovalRequest):
    result = run_agent(
        user_message=request.message,
        session_id=request.session_id,
        approved=True,)


    return ChatResponse(
        output = result["output"],
        pending_tool = result["pending_tool"],
        needs_approval = False,
        error = result["error"],
    )


# SECTION 7: REJECT ENDPOINT
@app.post("/reject", response_model=ChatResponse)
def reject(request: ApprovalRequest):
    result = run_agent(
        user_message=request.message,
        session_id=request.session_id,
        approved=False,)

    return ChatResponse(
        output = result["output"],
        pending_tool = result["pending_tool"],
        needs_approval = False,
        error = result["error"],
    )

# ENTRY POINT
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)