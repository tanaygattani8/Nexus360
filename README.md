# Nexus360 — Salesforce AI Agent

**🔗 Live demo: [nexus360-26bg.onrender.com](https://nexus360-26bg.onrender.com/)**
_(free tier, first load may take 30 to 60s to wake the server)_

A full-stack AI agent that reasons over CRM data, searches an internal knowledge base via RAG, and requires human approval before any write operation. It runs against a live Salesforce org or fully self-contained on local fallbacks.

Built to show the engineering behind production AI agents: safe tool use over real systems, retrieval quality you can measure, cost-aware model routing, and run analytics. That is the data-and-AI work of making agents grounded, measurable, and cheap to run. It also independently implements the primitives that platforms like Salesforce Agentforce and Data 360 productize (see [How this maps to Agentforce](#how-this-maps-to-agentforce-360)).

---

## What it does

- **Query live Salesforce data** in plain English — accounts, cases, opportunities
- **Search internal knowledge base** — policies, SLAs, playbooks via RAG (Qdrant)
- **Human-in-the-loop approval** — write operations pause; the server executes exactly the tool call it stashed when the agent proposed it, never the args the client echoes back and never a re-run of the agent
- **Run analytics** — a METRICS view (and `GET /analytics`) tracks tool mix, LLM calls skipped and cost avoided, human-approval rate, and p50/p95 latency: the core Agentforce Command Center signals, measured on your own runs
- **Measured retrieval** — an eval suite scores the *deployed* retrieval path (Precision@1/@3, MRR), judges answer quality on the agent's real answer, and tests abstention on out-of-scope questions
- **Persistent memory** — conversation history saved per session via Supabase
- **Full observability** — every agent run traced in LangSmith
- **React chat UI** — dark terminal aesthetic, approval cards with audit trail, session management, live backend health indicator
- **Expiry-proof fallbacks** — free-tier services die; the demo doesn't. No Supabase → local SQLite. No Qdrant Cloud → embedded local Qdrant. No Salesforce org → built-in mock data (`MOCK_SF=true` or automatic on connection failure). `/health` reports which mode each service is in.

---

## How this maps to Agentforce 360

This project was built from open-source primitives to understand, from the inside,
what a modern agent platform actually has to do. Each piece here corresponds to
something Salesforce ships as a product in Agentforce 360 / Data 360:

| Built here | Productized by Salesforce as | The idea |
|---|---|---|
| Binding approval: the server executes only the write it stashed, never the client's args | Einstein Trust Layer + human-in-the-loop guardrails | A deterministic trust boundary around a probabilistic model |
| Hybrid RAG + an eval suite (Precision@k, MRR, answer quality, abstention) | Data 360 retrievers + Agentforce Testing Center | Retrieval you measure, not retrieval you hope works |
| Template-vs-LLM routing (skip the LLM on deterministic turns) | Consumption pricing (Flex Credits, ~$0.10/action) rewards fewer calls | Cost and latency discipline as an engineering concern |
| Three-tier fallbacks + honest `/health` + `/analytics` | Command Center observability and reliability | Design for dependency failure; make system state visible |
| `_soql_escape`, arg sanitization, valid-stage enums as one source of truth | Guardrails and injection-safe actions | Tool args are attacker-influenced input |

The point is not that this replaces Agentforce. It is that building the primitives
is how you learn where the hard problems in agent systems actually live: evaluation,
grounding quality, cost, and observability.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`, override with `GROQ_MODEL`) |
| Agent framework | LangGraph + LangChain (langchain 1.3.1) |
| CRM | Salesforce (simple-salesforce) |
| Vector DB | Qdrant Cloud |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Keyword search | rank-bm25 (BM25Okapi) |
| Memory | Supabase (PostgreSQL) |
| Backend | Python 3.12, FastAPI |
| Frontend | React 19, TypeScript, Vite |
| Observability | LangSmith |

---

## Project Structure

```
nexus360/
├── backend/
│   ├── .env                    # credentials — never commit
│   ├── requirements.txt        # all Python dependencies
│   ├── tools.py                # 4 Salesforce @tool functions
│   ├── agent.py                # LangGraph graph — 7 nodes, dynamic prompts, token efficiency
│   ├── main.py                 # FastAPI — /chat /approve /reject /health
│   ├── memory.py               # Supabase conversation memory
│   ├── analytics.py            # run telemetry (tool mix, LLM-skip, approvals, latency) → /analytics
│   ├── knowledge_base.py       # Qdrant RAG — hybrid search + reranking + query rewriting
│   ├── eval.py                 # RAG eval — deployed-path retrieval, agent-answer quality, abstention
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
  ├── reason         → LLM decides which tool(s) to call
  ├── validate       → sanitize inputs, check EVERY tool call — any WRITE pauses the turn
  ├── approval_pause → WRITE ops end the run here; /approve executes exactly the
  │                    approved tool + args directly (the agent is never re-run)
  ├── execute        → ToolNode runs the Salesforce or RAG tool(s)
  ├── track_tools    → records which tools ran this turn
  ├── respond        → LLM formats output (or deterministic template for Salesforce reads;
  │                    or the reason node's reply directly when no tool was needed)
  └── save_memory    → persist turn to Supabase (or local SQLite fallback)
     ↓
FastAPI returns JSON { output, pending_tool, needs_approval, error }
     ↓
React renders answer OR approval card
```

### RAG Pipeline

```
User query
  → LLM rewrites query for better retrieval
  → Dense search (Qdrant cosine similarity)
  → BM25 keyword search (rank-bm25)
  → Reciprocal Rank Fusion (combines both rankings)
  → Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  → Top 3 results injected into LLM context
```

**Eval:** `python eval.py` scores the pipeline you actually deploy (BM25-lite when
`RAG_LITE=true`, full hybrid otherwise) rather than a pipeline that never runs in
production. It reports retrieval Precision@1/@3 and MRR, answer quality judged on
the agent's synthesized answer (not raw retrieved text), and an abstention rate on
out-of-scope questions the KB cannot answer. Run it to get numbers for your config.

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
| `/approve` | POST | Approve the pending write (executes the server-stashed call) |
| `/reject` | POST | Reject a pending write operation |
| `/health` | GET | Active mode per service (salesforce/memory/qdrant) |
| `/analytics` | GET | Aggregated run metrics for the METRICS dashboard |

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

## Eval

Run the RAG eval suite to measure retrieval and answer quality:
```bash
python eval.py
```

Outputs:
- Retrieval Precision@1, Precision@3, and MRR on the deployed retrieval path
- Answer quality (LLM-as-judge, 1-5) scored on the agent's real answer, over multiple runs so variance shows
- Abstention rate on out-of-scope questions the KB can't answer (safety signal)
- Weak spot detection — retrieval misses, low quality answers, and hallucinations on unknown questions

## Deploy — live demo on Render (free)

The whole app runs as **one Docker container**: FastAPI serves both the API and the
built React UI (same origin, no CORS). Thanks to the service fallbacks below, the
deployed demo needs exactly **one secret**: `GROQ_API_KEY`.

Render's free tier has 512MB RAM — too small for the PyTorch embedding/reranker
models — so the deploy sets `RAG_LITE=true`: retrieval falls back to BM25 + LLM
query rewrite (accurate on this 8-document KB), and torch is never installed.
The full hybrid pipeline (dense + RRF + cross-encoder rerank) still runs locally.

1. Create a free account at [render.com](https://render.com) (no credit card needed).
2. **New → Blueprint**, connect this GitHub repo — Render reads `render.yaml`
   and prompts you for `GROQ_API_KEY`.
   (Or: **New → Web Service** → Docker runtime → free plan → add env vars
   `RAG_LITE=true` and `GROQ_API_KEY` yourself.)
3. First build takes a few minutes; your shareable URL is
   `https://<service-name>.onrender.com` (this repo's is live at
   [nexus360-26bg.onrender.com](https://nexus360-26bg.onrender.com/)).

Notes: free services sleep after ~15 min idle and wake on the next visit
(~30-60s cold start — open the link once before sharing it). Storage is
ephemeral — SQLite chat memory resets on restart (fine for a demo). To run
against live Salesforce/Supabase instead of the fallbacks, add those env vars
in the Render dashboard.

A statically hosted frontend (e.g. Hugging Face Static Space serving
`frontend/dist`) also works: build with `VITE_API_URL=<backend-url>` and set
`ALLOWED_ORIGINS=<frontend-origin>` on the backend. The single-container
deploy above is simpler — one URL, no CORS.

## Resilience — free tiers expire, the demo doesn't

Every external service has a local fallback, chosen automatically at startup:

| Service | Primary | Fallback | Trigger |
|---|---|---|---|
| Salesforce | Live dev org | Built-in mock data (same seed accounts/cases/opps) | Connection fails, creds missing, or `MOCK_SF=true` |
| Memory | Supabase | Local SQLite (`backend/memory.db`) | `SUPABASE_URL` not set |
| Vector DB | Qdrant Cloud | Embedded local Qdrant (`backend/qdrant_local/`, auto-seeded) | `QDRANT_URL` not set |

`GET /health` reports the active mode per service, and the UI header shows an
honest LIVE/OFFLINE dot plus an `SF: MOCK` badge when Salesforce is mocked.
Only `GROQ_API_KEY` is strictly required.

## Known Limitations

- No authentication on API endpoints (intentional for local demo)
- No streaming responses (FastAPI + LangGraph both support it — next upgrade)
- Single reasoning pass per turn — parallel tool calls work, but the agent doesn't loop back for sequential multi-step tool chains (a ReAct loop would trade away the template-response token savings)
- If the LLM proposes two writes in one turn, only the first is surfaced for approval
- No fuzzy account name matching (RAG semantic search is the right fix — scoped to Phase 3+)

---

## Observability

Every agent run is traced in LangSmith at [smith.langchain.com](https://smith.langchain.com).
Traces show: full graph execution, node latency, token usage, tool inputs/outputs.
