# ── Imports ───────────────────────────────────────────────────────────────────
import os
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
)

from memory import save_message, load_history
from knowledge_base import search_knowledge_base as _search_kb

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


# SECTION 3 - PROMPTS

BASE_PROMPT = """You are Nexus360, an AI assistant with live access to Salesforce CRM data
and an internal knowledge base of company policies and playbooks.
 
Core rules:
- Use Salesforce tools for live account, case, and opportunity data.
- Use search_knowledge_base for policies, SLAs, playbooks, and internal processes.
- Use both tools when a question requires live data AND policy guidance.
- Always cite your source: (Source: Salesforce) or (Source: knowledge base, <doc title>).
- Be concise. Lead with the answer, follow with supporting detail."""

TOOL_CONTEXT = {
    "get_account_health": """
SALESFORCE ACCOUNT CONTEXT:
- AnnualRevenue is in USD.
- Open Cases count includes all non-Closed cases (New, In Progress, Escalated).
- Opportunity StageName values: Prospecting, Qualification, Needs Analysis,
  Value Proposition, Id. Decision Makers, Perception Analysis,
  Proposal/Price Quote, Negotiation/Review, Closed Won, Closed Lost.
- CloseDate is the expected contract close date, not a guarantee.
- Interpret account health: 0 open High cases = healthy, 1-2 = at risk, 3+ = critical.
- If AnnualRevenue is null, the account has not reported revenue.""",
 
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
 
    "update_opportunity_stage": """
SALESFORCE OPPORTUNITY STAGE CONTEXT:
- Valid stage progression: Prospecting → Qualification → Needs Analysis →
  Value Proposition → Id. Decision Makers → Perception Analysis →
  Proposal/Price Quote → Negotiation/Review → Closed Won / Closed Lost.
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
    Extracts the last ToolMessage content and returns it directly with a source tag.
    """
    # Find the last ToolMessage — that's the tool's raw output
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
 
    if not tool_messages:
        return {"final_output": "No tool result available."}
 
    tool_content = tool_messages[-1].content
    tools_used   = state.get("tools_used", [])
 
    # Determine source label
    write_tools = {"update_opportunity_stage", "create_support_case"}
    if any(t in write_tools for t in tools_used):
        source = "Salesforce (write confirmed)"
    else:
        source = "Salesforce"
 
    # Return the raw tool output with source tag appended
    # The tool output is already well-formatted plain text
    final = f"{tool_content.strip()}\n\n(Source: {source})"
 
    print(f"⚡ Template response used — skipped LLM call")
    return {
        "final_output": final,
        "messages":     [],   # no new AIMessage to add
    }


# SECTION 4 — LLM

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


# SECTION 4 — NODES
# Node 1: REASON
def reason(state: State) -> dict:
    messages = [SystemMessage(content=BASE_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    pending = None
    if response.tool_calls:
        tc      = response.tool_calls[0]
        pending = {"name": tc["name"], "args": tc["args"]}

    return {
        "messages":     [response],
        "pending_tool": pending,
    }


# Node 2: VALIDATE
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


# Node 3: HUMAN APPROVAL
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
    UPDATED — smart routing:
    - If LLM is genuinely needed (KB search or multi-tool): make the LLM call
    - Otherwise: use a deterministic template — saves 300-500 tokens per query
    """
    tools_used = state.get("tools_used", [])
 
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


# Node 6: SAVE MEMORY 
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


# SECTION 6 — ROUTING

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


# SECTION 7 — BUILD THE GRAPH

def build_graph():
    graph = StateGraph(State)

    graph.add_node("reason",         reason)
    graph.add_node("validate",       validate)
    graph.add_node("human_approval", human_approval)
    graph.add_node("execute",        tool_node)
    graph.add_node("track_tools", track_tools)
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

    graph.add_edge("execute",     "track_tools")
    graph.add_edge("track_tools", "respond")
    graph.add_edge("respond",     "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


# SECTION 8 — PUBLIC INTERFACE

def run_agent(
    user_message: str,
    session_id:   str       = "default",
    approved:     bool | None = None,
) -> dict:
    graph = build_graph()

    # Load conversation history from Supabase
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

    # Build initial state 
    initial_state: State = {
        "messages":     all_messages,
        "session_id":   session_id,
        "pending_tool": None,
        "approved":     approved,
        "final_output": "",
        "tools_used":   [],
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


# SMOKE TESTS  —  python agent.py

if __name__ == "__main__":
    TEST_SESSION = "token-test-001"
 
    tests = [
        # Pure Salesforce read → should use TEMPLATE (⚡), no LLM call
        ("Salesforce read — expect template", "What is the account health for Acme Corp?"),
        # KB query → should use LLM (🧠), summarisation needed
        ("KB query — expect LLM", "What is the SLA for high priority cases?"),
        # Combined → should use LLM (🧠), synthesis needed
        ("Combined — expect LLM", "What is the account health for Globex Inc and what does the escalation policy say I should do?"),
    ]
 
    for label, query in tests:
        print(f"\n{'═'*60}")
        print(f"TEST: {label}")
        print(f"QUERY: {query}")
        print('═'*60)
        r = run_agent(query, session_id=TEST_SESSION)
        if r["error"]:
            print(f"❌ Error: {r['error']}")
        else:
            print(f"✅ Output:\n{r['output']}")
