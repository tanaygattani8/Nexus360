# ── Imports ───────────────────────────────────────────────────────────────────
import os
from typing import Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from tools import (
    get_account_health,
    list_open_cases,
    update_opportunity_stage,
    create_support_case,
)

from memory import save_message, load_history
from knowledge_base import search_knowledge_base as _search_kb

load_dotenv()

import langchain
langchain.debug = False
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for policies, playbooks, and processes.
    Use this when asked about escalation policy, SLAs, renewal process,
    onboarding, opportunity stages, or data privacy rules.

    Args:
        query: The question or topic to search for.
    """
    return _search_kb(query)


READ_TOOLS  = [get_account_health, list_open_cases, search_knowledge_base]
WRITE_TOOLS = [update_opportunity_stage, create_support_case]
ALL_TOOLS   = READ_TOOLS + WRITE_TOOLS

WRITE_TOOL_NAMES = {t.name for t in WRITE_TOOLS}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STATE
# ══════════════════════════════════════════════════════════════════════════════
# load_memory node is REMOVED — history is now loaded in run_agent() directly
# before building initial_state. This guarantees correct message ordering:
# [history...] + [current message] → LLM always sees latest message last.

class State(TypedDict):
    messages:     Annotated[list, add_messages]
    session_id:   str
    pending_tool: dict | None
    approved:     bool | None
    final_output: str


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LLM
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Nexus360, an AI assistant with live access to Salesforce data
and an internal knowledge base of company policies and playbooks.

When answering:
- Use Salesforce tools for live account, case, and opportunity data.
- Use search_knowledge_base for questions about policies, SLAs, escalation procedures,
  renewal playbooks, onboarding, or internal processes.
- Use both if needed — e.g. check account health AND look up the renewal playbook.
- Be concise and always cite your source: (Source: Salesforce) or (Source: knowledge base, <doc title>).
"""


def _build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in .env")
    return ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )

llm            = _build_llm()
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NODES
# ══════════════════════════════════════════════════════════════════════════════

# ── Node 1: REASON ────────────────────────────────────────────────────────────
def reason(state: State) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    pending = None
    if response.tool_calls:
        tc      = response.tool_calls[0]
        pending = {"name": tc["name"], "args": tc["args"]}

    return {
        "messages":     [response],
        "pending_tool": pending,
    }


# ── Node 2: VALIDATE ──────────────────────────────────────────────────────────
def validate(state: State) -> dict:
    tool_call = state["pending_tool"]

    if not tool_call:
        return {"approved": True}

    tool_name = tool_call["name"]
    args      = tool_call["args"]

    # Input sanitization
    for key in ("account_name", "opportunity_name"):
        if key in args:
            args[key] = args[key].strip().rstrip(".,!?;:")

    # Already approved on re-run → respect it
    if state["approved"] is True:
        return {"approved": True, "pending_tool": {"name": tool_name, "args": args}}

    if tool_name in WRITE_TOOL_NAMES:
        print(f"\n⚠️  WRITE OPERATION DETECTED: {tool_name}")
        print(f"   Args: {args}")
        print(f"   → Routing to human approval\n")
        return {
            "approved":     None,
            "pending_tool": {"name": tool_name, "args": args},
        }

    return {"approved": True, "pending_tool": {"name": tool_name, "args": args}}


# ── Node 3: HUMAN APPROVAL ────────────────────────────────────────────────────
def human_approval(state: State) -> dict:
    tool_call = state["pending_tool"]
    print(f"\n🛑 HUMAN APPROVAL REQUIRED")
    print(f"   Tool:      {tool_call['name']}")
    print(f"   Arguments: {tool_call['args']}")
    print(f"   Approved:  {state['approved']}")

    if state["approved"] is None:
        print("   ⏸️  Waiting for human decision\n")
        return {"approved": None}

    if state["approved"] is True:
        print("   ✅ Approved — executing\n")
        return {"approved": True}

    print("   ❌ Rejected — action blocked\n")
    return {
        "approved":     False,
        "final_output": f"Action blocked. '{tool_call['name']}' requires approval before it can run.",
    }


# ── Node 4: EXECUTE ───────────────────────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)


# ── Node 5: RESPOND ───────────────────────────────────────────────────────────
def respond(state: State) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages":     [response],
        "final_output": response.content,
    }


# ── Node 6: SAVE MEMORY ───────────────────────────────────────────────────────
# Saves ONLY the current turn — not the full history (that's already in Supabase).
# We identify the current user message as the last HumanMessage in state.

def save_memory(state: State) -> dict:
    session_id = state.get("session_id", "default")

    # The current user message is the last HumanMessage in state
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if human_messages:
        current_user_msg = human_messages[-1].content
        save_message(session_id, "user", current_user_msg)

    # Save agent's final answer
    if state.get("final_output"):
        save_message(session_id, "assistant", state["final_output"])
        print(f"💾 Saved turn to memory for session '{session_id}'")

    return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def after_reason(state: State) -> Literal["validate", "respond"]:
    if state["pending_tool"]:
        return "validate"
    return "respond"


def after_validate(state: State) -> Literal["human_approval", "execute"]:
    if state["approved"] is None:
        return "human_approval"
    return "execute"


def after_approval(state: State) -> Literal["execute", "__end__"]:
    if state["approved"] is True:
        return "execute"
    return "__end__"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
# load_memory is no longer a node — history is loaded in run_agent() before
# the graph starts. Entry point is now "reason" directly.

def build_graph():
    graph = StateGraph(State)

    graph.add_node("reason",         reason)
    graph.add_node("validate",       validate)
    graph.add_node("human_approval", human_approval)
    graph.add_node("execute",        tool_node)
    graph.add_node("respond",        respond)
    graph.add_node("save_memory",    save_memory)

    graph.set_entry_point("reason")

    graph.add_conditional_edges("reason", after_reason, {
        "validate": "validate",
        "respond":  "respond",
    })

    graph.add_conditional_edges("validate", after_validate, {
        "human_approval": "human_approval",
        "execute":        "execute",
    })

    graph.add_conditional_edges("human_approval", after_approval, {
        "execute":  "execute",
        "__end__":  END,
    })

    graph.add_edge("execute",     "respond")
    graph.add_edge("respond",     "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
# History is loaded HERE before the graph runs — guarantees correct order:
# [old msg 1, old msg 2, ..., current message]
# The LLM always sees the current message last, which is what it expects.

def run_agent(
    user_message: str,
    session_id:   str       = "default",
    approved:     bool | None = None,
) -> dict:
    graph = build_graph()

    # ── Load conversation history from Supabase ────────────────────────────────
    history     = load_history(session_id, limit=10)
    all_messages = []

    for msg in history:
        if msg["role"] == "user":
            all_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            all_messages.append(AIMessage(content=msg["content"]))

    # Current message goes LAST — after all history
    all_messages.append(HumanMessage(content=user_message))

    if len(all_messages) > 1:
        print(f"📚 Loaded {len(all_messages) - 1} messages from memory for session '{session_id}'")

    # ── Build initial state ────────────────────────────────────────────────────
    initial_state: State = {
        "messages":     all_messages,
        "session_id":   session_id,
        "pending_tool": None,
        "approved":     approved,
        "final_output": "",
    }

    try:
        result = graph.invoke(initial_state)
        return {
            "output":       result.get("final_output", ""),
            "pending_tool": result.get("pending_tool"),
            "approved":     result.get("approved"),
            "error":        None,
        }
    except Exception as exc:
        return {
            "output":       "",
            "pending_tool": None,
            "approved":     None,
            "error":        str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TESTS  —  python agent.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_SESSION = "smoke-test-002"

    print("\n" + "═"*60)
    print("TEST 1: RAG query")
    print("═"*60)
    r1 = run_agent("What is the SLA for high priority cases?", session_id=TEST_SESSION)
    print(f"✅ {r1['output']}" if not r1["error"] else f"❌ {r1['error']}")

    print("\n" + "═"*60)
    print("TEST 2: Memory — should reference previous question")
    print("═"*60)
    r2 = run_agent("What was my previous question?", session_id=TEST_SESSION)
    print(f"✅ {r2['output']}" if not r2["error"] else f"❌ {r2['error']}")

    print("\n" + "═"*60)
    print("TEST 3: Salesforce read")
    print("═"*60)
    r3 = run_agent("What is the account health for Acme Corp?", session_id=TEST_SESSION)
    print(f"✅ {r3['output']}" if not r3["error"] else f"❌ {r3['error']}")
