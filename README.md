# Nexus360 — Salesforce AI Agent

A production-grade AI agent that connects to a live Salesforce org, reasons over CRM data, searches an internal knowledge base via RAG, and requires human approval before any write operation.

Built as a demonstration of end-to-end agent architecture for the Salesforce Agentforce Engineer role.

---

## What it does

- **Query live Salesforce data** in plain English — accounts, cases, opportunities
- **Search internal knowledge base** — policies, SLAs, playbooks via RAG (Qdrant)
- **Human-in-the-loop approval** — write operations pause and require confirmation before executing
- **Persistent memory** — conversation history saved per session via Supabase
- **Full observability** — every agent run traced in LangSmith
- **React chat UI** — dark terminal aesthetic, approval cards, session management

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) |
| Agent framework | LangGraph 1.2 + LangChain 1.3 |
| CRM | Salesforce (simple-salesforce) |
| Vector DB | Qdrant Cloud |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Memory | Supabase (PostgreSQL) |
| Backend | Python 3.12, FastAPI |
| Frontend | React 18, TypeScript, Vite |
| Observability | LangSmith |

---

## Project Structure

```
nexus360/
├── backend/
│   ├── .env                    # credentials — never commit
│   ├── requirements.txt        # all Python dependencies
│   ├── tools.py                # 4 Salesforce @tool functions
│   ├── agent.py                # LangGraph graph — 6 nodes
│   ├── main.py                 # FastAPI — /chat /approve /reject /health
│   ├── memory.py               # Supabase conversation memory
│   ├── knowledge_base.py       # Qdrant RAG — setup, seed, search
│   ├── seed_data.py            # Salesforce test data (8 accounts/opps/cases)
│   ├── test_connections.py     # verify Salesforce + Groq connections
│   ├── test_tools.py           # verify all 4 tools against live SF data
│   └── test_phase3_connections.py  # verify Supabase + Qdrant connections
└── frontend/
    ├── src/
    │   ├── App.tsx             # chat UI — messages, approval flow, session mgmt
    │   └── index.css           # dark terminal styling
    ├── package.json
    └── vite.config.ts
```

---

## Setup

### Prerequisites
- Python 3.12+
- Node 18+
- Salesforce Developer Org (free at developer.salesforce.com)
- Groq API key (free at console.groq.com)
- Supabase project (free at supabase.com)
- Qdrant Cloud cluster (free at cloud.qdrant.io)
- LangSmith account (free at smith.langchain.com)

### Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

Create `backend/.env`:
```
# Groq
GROQ_API_KEY=your-groq-api-key

# Salesforce
SALESFORCE_USERNAME=your-username
SALESFORCE_PASSWORD=your-password
SALESFORCE_SECURITY_TOKEN=your-security-token
SALESFORCE_DOMAIN=login

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-publishable-key

# Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=nexus360
```

Create the Supabase conversations table (run in Supabase SQL editor):
```sql
CREATE TABLE conversations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Seed data and knowledge base:
```bash
python seed_data.py          # creates 8 accounts, opportunities, cases in Salesforce
python knowledge_base.py     # embeds 8 internal docs into Qdrant
```

Start the backend:
```bash
python main.py
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:5173
```

---

## Agent Architecture

```
User message
     ↓
FastAPI /chat (+ session_id)
     ↓
run_agent()
  └── Load history from Supabase (last 10 messages)
  └── Append current message
     ↓
LangGraph Graph
  ├── reason        → LLM decides which tool to call
  ├── validate      → sanitize inputs, classify READ vs WRITE
  ├── human_approval→ WRITE ops pause here, wait for /approve or /reject
  ├── execute       → ToolNode runs the Salesforce or RAG tool
  ├── respond       → LLM formats tool output into clean answer
  └── save_memory   → persist turn to Supabase
     ↓
FastAPI returns JSON { output, pending_tool, needs_approval, error }
     ↓
React renders answer OR approval card
```

### Tools

| Tool | Type | Description |
|---|---|---|
| `get_account_health` | READ | Account summary — revenue, cases, opportunities |
| `list_open_cases` | READ | All open cases for an account |
| `search_knowledge_base` | READ | RAG search — policies, SLAs, playbooks |
| `update_opportunity_stage` | WRITE | Move a deal to a new pipeline stage |
| `create_support_case` | WRITE | Log a new support case in Salesforce |

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/chat` | POST | Send a message, get a response or approval request |
| `/approve` | POST | Approve a pending write operation |
| `/reject` | POST | Reject a pending write operation |

---

## The 4-Phase Story

This project was built in 4 deliberate phases to demonstrate progressive improvement in agent design:

| Phase | What it adds | Why it matters |
|---|---|---|
| Foundation | Salesforce connection, 4 tools, seed data | Proves real integration, not mock data |
| Phase 1 | Naive LangGraph agent, FastAPI, React UI | Baseline agent — works but brittle and unsafe |
| Phase 2 | Custom graph, guardrails, human approval | Prevents unsafe writes, adds input sanitization |
| Phase 3 | Supabase memory, Qdrant RAG, LangSmith | Stateful, knowledge-grounded, observable |
| Phase 4 | README, architecture diagram, interview prep | Production-ready documentation |

---

## Known Limitations

- No authentication on API endpoints (intentional for local demo)
- No streaming responses (FastAPI + LangGraph both support it — next upgrade)
- Approval state held in React state (would use Supabase in production)
- No fuzzy account name matching (RAG semantic search is the right fix — scoped to Phase 3+)

---

## Observability

Every agent run is traced in LangSmith at [smith.langchain.com](https://smith.langchain.com).
Traces show: full graph execution, node latency, token usage, tool inputs/outputs.
