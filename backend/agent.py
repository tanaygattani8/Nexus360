# ── Imports ───────────────────────────────────────────────────────────────────
import os
import time
from typing import Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from tools import (
    get_account_health,
    list_open_cases,
    update_opportunity_stage,
    create_support_case,
    VALID_STAGES,
)

from memory import save_message, load_history
from knowledge_base import search_knowledge_base as _search_kb
from analytics import log_run

load_dotenv()

import langchain
langchain.debug = False

# SECTION 1 — TOOLS

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


# SECTION 2 — STATE

class State(TypedDict):
    messages:     Annotated[list, add_messages]
    session_id:   str
    pending_tool: dict | None
    approved:     bool | None
    final_output: str
    tools_used:   list[str]


# SECTION 3 — PROMPTS

BASE_PROMPT = """You are Nexus360, an AI assistant with live access to Salesforce CRM data
and an internal knowledge base of company policies and playbooks.

Core rules:
- Use Salesforce tools for live account, case, and opportunity data.
- Use search_knowledge_base for policies, SLAs, playbooks, and internal processes.
- Use both tools when a question requires live data AND policy guidance.
- Always cite your source: (Source: Salesforce) or (Source: knowledge base, <doc title>).
- Be concise. Lead with the answer, follow with supporting detail.
- Write in plain text. Do not use markdown formatting like ** or ##."""

_STAGES_LINE = ", ".join(VALID_STAGES)

TOOL_CONTEXT = {
    "get_account_health": f"""
SALESFORCE ACCOUNT CONTEXT:
- AnnualRevenue is in USD.
- Open Cases count includes all non-Closed cases (New, In Progress, Escalated).
- Opportunity StageName values: {_STAGES_LINE}.
- CloseDate is the expected contract close date, not a guarantee.
- Interpret account health: 0 open High cases = healthy, 1-2 = at risk, 3+ = critical.
- If AnnualRevenue is N/A, the account has not reported revenue.""",

    "list_open_cases": """
SALESFORCE CASE CONTEXT:
- Priority levels: High (production impact), Medium (workaround exists), Low (cosmetic/question).
- Status flow: New → In Progress → Escalated → Waiting on Customer → Closed.
- High priority cases open >24h should be escalated per policy.
- CreatedDate is in UTC. Convert to local time when communicating to users.
- If no open cases found, the account is in good standing on support.""",

    "search_knowledge_base": """
KNOWLEDGE BASE CONTEXT:
- Documents cover: escalation policy, SLA definitions, account health scoring,
  renewal playbook, onboarding checklist, support case best practices,
  opportunity stage definitions, data privacy policy.
- Relevance scores: above +3 = high confidence, 0 to +3 = moderate, below 0 = weak match.
- If the retrieved document has a weak relevance score, caveat your answer.
- Always name the specific document you are drawing from.""",

    "update_opportunity_stage": f"""
SALESFORCE OPPORTUNITY STAGE CONTEXT:
- Valid stage progression: {_STAGES_LINE.replace(', ', ' → ')}.
- Skipping stages (e.g. Qualification → Closed Won) flags a data quality issue.
- Closed Lost requires a Loss Reason to be filled in the record.
- Stage changes are immediately visible in Salesforce pipeline reports.""",

    "create_support_case": """
SALESFORCE CASE CREATION CONTEXT:
- Priority must match business impact: High = production down or data loss risk,
  Medium = broken with workaround, Low = cosmetic or question.
- Subject line should be specific: 'Login failure after SSO update' not 'Login issue'.
- New cases start in 'New' status and must be moved to 'In Progress' within 2 hours.
- The case is immediately visible to the account's CSM in Salesforce.""",
}

def build_system_prompt(tools_used: list[str]) -> str:
    """
    Build a dynamic system prompt by combining the base prompt with
    context blocks for each tool that was called this turn.

    Args:
        tools_used: List of tool names called during this turn.

    Returns:
        Complete system prompt string.
    """
    prompt = BASE_PROMPT

    # Inject context for each tool that was used
    for tool_name in set(tools_used):
        if tool_name in TOOL_CONTEXT:
            prompt += f"\n{TOOL_CONTEXT[tool_name]}"

    return prompt


# SECTION 4 — TOKEN EFFICIENCY HELPERS

def _needs_llm_response(tools_used: list[str]) -> bool:
    """
    Returns True only when the LLM is genuinely needed to format the response.
    Returns False when a deterministic template is sufficient.
    """
    # KB search always needs LLM — policy docs need summarisation
    if "search_knowledge_base" in tools_used:
        return True
    # Multiple different tools → synthesis required
    if len(set(tools_used)) > 1:
        return True
    # Pure Salesforce reads and write confirmations → use template
    return False


def _format_template_response(state: State) -> dict:
    """
    Format a response using a deterministic template when LLM is not needed.
    Joins ALL tool results from this turn (the same tool may have run more
    than once, e.g. account health for two accounts) and adds a source tag.
    """
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

    if not tool_messages:
        return {"final_output": "No tool result available."}

    body       = "\n\n".join(str(m.content).strip() for m in tool_messages)
    tools_used = state.get("tools_used", [])

    if any(t in WRITE_TOOL_NAMES for t in tools_used):
        source = "Salesforce (write confirmed)"
    else:
        source = "Salesforce"

    final = f"{body}\n\n(Source: {source})"

    print(f"⚡ Template response used — skipped LLM call")
    return {
        "final_output": final,
        "messages":     [],   # no new AIMessage to add
    }


# SECTION 5 — LLM

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


# SECTION 6 — NODES

# Node 1: REASON
def reason(state: State) -> dict:
    messages = [SystemMessage(content=BASE_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Node 2: VALIDATE
def validate(state: State) -> dict:
    """
    Sanitize tool arguments and gate write operations.
    Checks EVERY tool call in the turn — if any of them is a write,
    the whole turn pauses for human approval. (Checking only the first
    call would let a write sneak through alongside a read.)
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    # Input sanitization — mutate the message's tool_calls in place,
    # because ToolNode executes from the message itself.
    for tc in tool_calls:
        for key in ("account_name", "opportunity_name"):
            value = tc["args"].get(key)
            if isinstance(value, str):
                tc["args"][key] = value.strip().rstrip(".,!?;:")

    writes = [tc for tc in tool_calls if tc["name"] in WRITE_TOOL_NAMES]

    if writes:
        # ponytail: only the first write is surfaced for approval; the LLM
        # emitting two writes in one turn is rare — split into per-write
        # approval cards if it ever happens in practice.
        tc = writes[0]
        print(f"\n⚠️  WRITE OPERATION DETECTED: {tc['name']}")
        print(f"   Args: {tc['args']}")
        print(f"   → Pausing for human approval\n")
        return {
            "approved":     None,
            "pending_tool": {"name": tc["name"], "args": tc["args"]},
        }

    return {"approved": True, "pending_tool": None}


# Node 3: APPROVAL PAUSE
def approval_pause(state: State) -> dict:
    """
    A write needs approval. End this run and hand the pending tool back to
    the API. /approve executes EXACTLY this tool with EXACTLY these args —
    the agent is never re-run, so what the user approves is what executes.
    """
    tc = state["pending_tool"]
    print(f"\n🛑 HUMAN APPROVAL REQUIRED")
    print(f"   Tool:      {tc['name']}")
    print(f"   Arguments: {tc['args']}\n")
    return {"final_output": f"Approval required before running '{tc['name']}'."}


# Node 4: EXECUTE
tool_node = ToolNode(ALL_TOOLS)

# Node 5: TRACK TOOLS
def track_tools(state: State) -> dict:
    """
    Scan the message history for tool calls made this turn
    and record their names in state["tools_used"].
    """
    tools_used = []
    for msg in state["messages"]:
        # AIMessage with tool_calls = the LLM decided to call a tool
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["name"])

    print(f"🔧 Tools used this turn: {tools_used}")
    return {"tools_used": tools_used}


# Node 6: RESPOND
def respond(state: State) -> dict:
    """
    Smart routing:
    - No tools ran → the reason node's reply IS the answer (e.g. greetings)
    - KB search or multi-tool → LLM synthesis with dynamic prompt
    - Pure Salesforce reads/writes → deterministic template, saves an LLM call
    """
    tools_used = state.get("tools_used", [])

    if not tools_used:
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        return {"final_output": last_ai.content if last_ai else ""}

    if _needs_llm_response(tools_used):
        print(f"🧠 LLM response — tools require synthesis: {tools_used}")
        system_prompt = build_system_prompt(tools_used)
        messages      = [SystemMessage(content=system_prompt)] + state["messages"]
        response      = llm.invoke(messages)
        return {
            "messages":     [response],
            "final_output": response.content,
        }
    else:
        return _format_template_response(state)


# Node 7: SAVE MEMORY
def save_memory(state: State) -> dict:
    session_id = state.get("session_id", "default")

    # The current user message is the last HumanMessage in state
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if human_messages:
        save_message(session_id, "user", human_messages[-1].content)

    # Save agent's final answer
    if state.get("final_output"):
        save_message(session_id, "assistant", state["final_output"])
        print(f"💾 Saved turn to memory for session '{session_id}'")

    return {}


# SECTION 7 — ROUTING

def after_reason(state: State) -> Literal["validate", "respond"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "validate"
    return "respond"


def after_validate(state: State) -> Literal["approval_pause", "execute"]:
    if state["approved"] is None:
        return "approval_pause"
    return "execute"


# SECTION 8 — BUILD THE GRAPH

def build_graph():
    graph = StateGraph(State)

    graph.add_node("reason",         reason)
    graph.add_node("validate",       validate)
    graph.add_node("approval_pause", approval_pause)
    graph.add_node("execute",        tool_node)
    graph.add_node("track_tools",    track_tools)
    graph.add_node("respond",        respond)
    graph.add_node("save_memory",    save_memory)

    graph.set_entry_point("reason")

    graph.add_conditional_edges("reason", after_reason, {
        "validate": "validate",
        "respond":  "respond",
    })

    graph.add_conditional_edges("validate", after_validate, {
        "approval_pause": "approval_pause",
        "execute":        "execute",
    })

    # Approval pause ends the run — /approve executes the tool directly.
    graph.add_edge("approval_pause", END)

    graph.add_edge("execute",     "track_tools")
    graph.add_edge("track_tools", "respond")
    graph.add_edge("respond",     "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# Compiled once at import — not per request.
agent_graph = build_graph()


# SECTION 9 — PUBLIC INTERFACE

def run_agent(user_message: str, session_id: str = "default") -> dict:
    # Load conversation history from memory
    history      = load_history(session_id, limit=10)
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

    initial_state: State = {
        "messages":     all_messages,
        "session_id":   session_id,
        "pending_tool": None,
        "approved":     None,
        "final_output": "",
        "tools_used":   [],
    }

    try:
        started = time.perf_counter()
        result  = agent_graph.invoke(initial_state)
        latency_ms = (time.perf_counter() - started) * 1000

        tools_used = result.get("tools_used", [])
        pending    = result.get("pending_tool")
        if pending:
            path = "approval_pending"
        elif not tools_used:
            path = "direct"
        elif _needs_llm_response(tools_used):
            path = "llm"
        else:
            path = "template"
        log_run(session_id, tools_used, path, latency_ms)

        return {
            "output":       result.get("final_output", ""),
            "pending_tool": pending,
            "error":        None,
        }
    except Exception as exc:
        return {
            "output":       "",
            "pending_tool": None,
            "error":        str(exc),
        }


# SMOKE TESTS  —  python agent.py

if __name__ == "__main__":
    TEST_SESSION = "token-test-001"

    tests = [
        # No tool needed → reason node's reply is the answer
        ("Chitchat — expect direct reply", "Hello! What can you do?"),
        # Pure Salesforce read → should use TEMPLATE (⚡), no LLM call
        ("Salesforce read — expect template", "What is the account health for Acme Corp?"),
        # KB query → should use LLM (🧠), summarisation needed
        ("KB query — expect LLM", "What is the SLA for high priority cases?"),
        # Combined → should use LLM (🧠), synthesis needed
        ("Combined — expect LLM", "What is the account health for Globex Inc and what does the escalation policy say I should do?"),
        # Write → should pause with pending_tool, nothing executed
        ("Write — expect approval pause", "Move the Acme Corp - Enterprise License deal to Closed Won"),
    ]

    for label, query in tests:
        print(f"\n{'═'*60}")
        print(f"TEST: {label}")
        print(f"QUERY: {query}")
        print('═'*60)
        r = run_agent(query, session_id=TEST_SESSION)
        if r["error"]:
            print(f"❌ Error: {r['error']}")
        elif r["pending_tool"]:
            print(f"⏸ Approval required: {r['pending_tool']}")
        else:
            print(f"✅ Output:\n{r['output']}")
