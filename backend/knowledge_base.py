# ── Imports ───────────────────────────────────────────────────────────────────
import os
import math
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from config import GROQ_MODEL

load_dotenv()

# RAG_LITE=true skips torch + the embedding/reranker models entirely and
# retrieves with BM25 only — for hosts with <1GB RAM (e.g. Render free tier).
# With 8 documents, BM25 + LLM query rewrite retrieves accurately; the full
# hybrid pipeline (dense + RRF + rerank) runs wherever RAM allows.
RAG_LITE = os.getenv("RAG_LITE", "").lower() == "true"

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME  = "nexus360_knowledge"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # small, fast, 384 dimensions
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"

VECTOR_SIZE      = 384
TOP_K_RETRIEVAL  = 6
TOP_K_FINAL      = 3
RERANK_THRESHOLD = 0.0 # can be negative
# Cap on how many fused candidates reach the cross-encoder. Cross-encoders are
# expensive (one model pass per candidate), so at scale you fuse wide then rerank
# only the fused top-N — and RRF's ordering is what decides which candidates
# survive that cut. A no-op at 8 docs (fused list is always smaller), but it
# makes the retrieve → fuse → rerank pipeline correct as the corpus grows.
RERANK_CANDIDATES = 25


# ── Clients & Models───────────────────────────────────────────────────────────
# Qdrant Cloud when configured, otherwise a local embedded Qdrant (same API,
# stored on disk, never expires). If your free cloud cluster gets suspended,
# just remove QDRANT_URL from .env and everything keeps working.

if RAG_LITE:
    QDRANT_MODE = "bm25-lite"   # vectors unused; /health reports it honestly
elif os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY"):
    QDRANT_MODE = "cloud"
else:
    QDRANT_MODE = "local"
_LOCAL_QDRANT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_local")

_qdrant: QdrantClient | None = None  # created once, reused across calls

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if QDRANT_MODE == "cloud":
            _qdrant = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )
        else:
            print(f"[qdrant] No cloud credentials — using local embedded Qdrant at {_LOCAL_QDRANT_PATH}")
            _qdrant = QdrantClient(path=_LOCAL_QDRANT_PATH)
            # Local mode is zero-setup: seed the collection on first use
            existing = [c.name for c in _qdrant.get_collections().collections]
            if COLLECTION_NAME not in existing:
                setup_collection()
                seed_documents()
    return _qdrant

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")
    return ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=128,  #query rewriting needs very few tokens
    )

if RAG_LITE:
    print("[kb] RAG_LITE=true — BM25-only retrieval, skipping embedding/reranker models")
    embedder = None
    reranker = None
else:
    # Imported here so RAG_LITE hosts never need torch installed at all
    from sentence_transformers import SentenceTransformer, CrossEncoder

    # SentenceTransformer downloads the model on first run (~90MB, cached after)
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print("[kb] Embedding model loaded")

    print("Loading reranker model...")
    reranker = CrossEncoder(RERANK_MODEL)
    print("[kb] Reranker loaded")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 - DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════


DOCUMENTS = [
    {
        "id":       1,
        "title":    "Customer Escalation Policy",
        "category": "policy",
        "content":  """
        Escalation Policy — Customer Success
        When to escalate a support case:
        - Any High priority case open for more than 24 hours must be escalated to the account manager.
        - Any case involving data loss or security must be escalated immediately to the engineering lead.
        - Cases with 3 or more reopen events should be flagged for a root cause analysis.
        Escalation steps:
        1. Tag the case as 'Escalated' in Salesforce.
        2. Notify the account manager via Slack within 1 hour.
        3. Schedule a 30-minute call with the customer within 4 business hours.
        4. Post an update in the #escalations channel with case ID, customer name, and issue summary.
        """
    },
    {
        "id":       2,
        "title":    "SLA Definitions",
        "category": "policy",
        "content":  """
        Service Level Agreement (SLA) — Response Times
        High Priority cases:
        - First response: within 2 business hours
        - Resolution target: within 24 business hours
        - Escalation if breached: immediate notification to VP of Customer Success
        Medium Priority cases:
        - First response: within 8 business hours
        - Resolution target: within 72 business hours
        Low Priority cases:
        - First response: within 24 business hours
        - Resolution target: within 7 business days
        SLA clock starts when the case status changes from New to In Progress.
        SLA is paused when status is set to Waiting on Customer.
        """
    },
    {
        "id":       3,
        "title":    "Account Health Scoring",
        "category": "playbook",
        "content":  """
        Account Health Scoring Guide
        Health scores are calculated monthly and visible in Salesforce.
        Green (Healthy): 
        - No open High priority cases
        - At least one Closed Won opportunity in the last 12 months
        - NPS score above 7
        Amber (At Risk):
        - 1-2 open High priority cases
        - No new opportunities in the last 6 months
        - NPS score between 4 and 7
        Red (Critical):
        - 3 or more open High priority cases
        - No activity in the last 90 days
        - NPS score below 4
        Recommended action for Red accounts: trigger the retention workflow within 48 hours.
        """
    },
    {
        "id":       4,
        "title":    "Renewal Playbook",
        "category": "playbook",
        "content":  """
        Renewal Playbook — Account Managers
        90 days before renewal:
        - Review account health score in Salesforce
        - Schedule an Executive Business Review (EBR) call
        - Identify upsell opportunities based on usage data
        60 days before renewal:
        - Send renewal proposal
        - Loop in Solutions Engineer if expansion is likely
        30 days before renewal:
        - Daily check on contract status
        - Escalate to VP if no signed contract yet
        - Offer early renewal discount (up to 10%) with manager approval
        If renewal is at risk:
        - Trigger retention workflow in Nexus360
        - Open a High priority case tagged 'Renewal Risk'
        - Notify the VP of Customer Success immediately
        """
    },
    {
        "id":       5,
        "title":    "Onboarding Checklist",
        "category": "process",
        "content":  """
        New Customer Onboarding Checklist
        Week 1:
        - Send welcome email with login credentials and onboarding guide
        - Schedule kickoff call within 3 business days
        - Create Salesforce account and all contacts
        - Assign dedicated Customer Success Manager (CSM)
        Week 2-4:
        - Complete product training sessions (minimum 3 sessions)
        - Ensure customer has submitted at least one support case (proves they know how)
        - Confirm integration with customer's existing tools
        Day 30 check-in:
        - Survey customer satisfaction (target NPS > 8 for new customers)
        - Review any open cases and resolve blockers
        - Set 90-day success milestones
        Escalate to CSM lead if kickoff call not completed within 5 business days.
        """
    },
    {
        "id":       6,
        "title":    "Support Case Best Practices",
        "category": "process",
        "content":  """
        Support Case Best Practices
        When creating a case:
        - Always link the case to the correct Salesforce Account
        - Subject line must be specific: 'Login failure after 2FA update' not 'Login issue'
        - Description must include: steps to reproduce, expected vs actual behavior, impact
        - Set priority based on business impact, not customer urgency
        Priority guide:
        - High: production system down, data loss risk, security issue
        - Medium: feature broken but workaround exists
        - Low: cosmetic issue, feature request, question
        Case hygiene:
        - Update case status daily if active
        - Never leave a case in 'New' status for more than 2 hours
        - Always add internal notes when handing off to another team member
        """
    },
    {
        "id":       7,
        "title":    "Opportunity Stage Definitions",
        "category": "sales",
        "content":  """
        Opportunity Stage Definitions — Sales Team
        Prospecting: Initial contact made, need identified. Probability: 10%
        Qualification: Budget, authority, need, and timeline confirmed (BANT). Probability: 25%
        Proposal: Formal proposal sent and reviewed. Probability: 60%
        Negotiation: Contract in legal review or active negotiation. Probability: 90%
        Closed Won: Contract signed. Probability: 100%
        Closed Lost: Deal lost. Always fill in the Loss Reason field.
        These are the only valid stages in our Salesforce org.
        Do not skip stages — moving from Qualification directly to Closed Won flags a data quality issue.
        """
    },
    {
        "id":       8,
        "title":    "Data Privacy and Compliance",
        "category": "policy",
        "content":  """
        Data Privacy and Compliance Policy
        Customer data handling:
        - Never share customer data outside of approved tools (Salesforce, Supabase, internal Slack)
        - All exports of customer data require manager approval
        - PII (names, emails, phone numbers) must not be pasted into public tools or AI assistants
        Salesforce data rules:
        - Do not bulk delete records without a full backup
        - All mass updates (10+ records) require a second approval from team lead
        - Audit logs are reviewed quarterly by the compliance team
        AI tool usage:
        - AI assistants (including Nexus360) must not be used to process PII without consent
        - All AI-generated content sent to customers must be reviewed by a human before sending
        - Log all AI actions in the audit trail — Nexus360 does this automatically
        """
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# SETUP COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
# Creates the Qdrant collection if it doesn't exist.
# A collection is like a table — it holds all the document vectors.

def setup_collection() -> None:
    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,   # cosine similarity — standard for text
        ),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")


# ══════════════════════════════════════════════════════════════════════════════
# SEED DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
# Embeds all documents and upserts them into Qdrant.
# Upsert = insert if new, update if ID already exists — safe to run multiple times.

def seed_documents() -> None:
    client = _get_qdrant()

    print(f"Embedding {len(DOCUMENTS)} documents...")
    texts  = [doc["content"] for doc in DOCUMENTS]
    vectors = embedder.encode(texts, show_progress_bar=True).tolist()

    points = [
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload={
                "title":    doc["title"],
                "category": doc["category"],
                "content":  doc["content"],
            },
        )
        for doc, vector in zip(DOCUMENTS, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ {len(points)} documents upserted into Qdrant")


# ══════════════════════════════════════════════════════════════════════════════
# QUERY REWRITING
# ══════════════════════════════════════════════════════════════════════════════
def rewrite_query(query: str) -> str:
    """
    Rewrite a natural language query into a precise search query
    optimised for retrieving relevant policy and playbook documents.
    """
    llm = _get_llm()
    messages = [
        SystemMessage(content="""You are a search query optimizer.
Rewrite the user's question into a precise, keyword-rich search query
for retrieving relevant documents from a CRM knowledge base.
The knowledge base contains: escalation policies, SLA definitions,
account health scoring, renewal playbooks, onboarding checklists,
support case best practices, opportunity stage definitions, data privacy policies.
Return ONLY the rewritten query. No explanation. No punctuation at the end."""),
        HumanMessage(content=f"Rewrite this query: {query}")
    ]
    response = llm.invoke(messages)
    rewritten = response.content.strip()
    print(f"🔄 Query rewritten: '{query}' → '{rewritten}'")
    return rewritten


# BM25 INDEX
_bm25_corpus   = [doc["content"].lower().split() for doc in DOCUMENTS]
_bm25_index    = BM25Okapi(_bm25_corpus)
_bm25_doc_ids  = [doc["id"] for doc in DOCUMENTS]
 
 
def _bm25_search(query: str, top_k: int) -> list[dict]:
    """
    BM25 keyword search over the local document corpus.
    Returns top_k documents with their BM25 scores.
    """
    tokens  = query.lower().split()
    scores  = _bm25_index.get_scores(tokens)
 
    # Pair each doc with its score and sort descending
    ranked  = sorted(
        zip(_bm25_doc_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
 
    results = []
    for doc_id, score in ranked:
        doc = next(d for d in DOCUMENTS if d["id"] == doc_id)
        results.append({
            "id":      doc_id,
            "title":   doc["title"],
            "content": doc["content"],
            "score":   float(score),
            "source":  "bm25",
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DENSE SEARCH (Qdrant)
# ══════════════════════════════════════════════════════════════════════════════
 
def _dense_search(query: str, top_k: int) -> list[dict]:
    """
    Dense vector search using Qdrant + cosine similarity.
    """
    client = _get_qdrant()
    vector = embedder.encode(query).tolist()
 
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    ).points
 
    return [
        {
            "id":      hit.id,
            "title":   hit.payload["title"],
            "content": hit.payload["content"],
            "score":   hit.score,
            "source":  "dense",
        }
        for hit in results
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RECIPROCAL RANK FUSION
# ══════════════════════════════════════════════════════════════════════════════
def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results:  list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion.
    Returns a deduplicated, re-ranked list of documents.
    """
    scores: dict[int, float] = {}
    docs:   dict[int, dict]  = {}
 
    for rank, doc in enumerate(dense_results):
        doc_id          = doc["id"]
        scores[doc_id]  = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id]    = doc
 
    for rank, doc in enumerate(bm25_results):
        doc_id          = doc["id"]
        scores[doc_id]  = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        docs[doc_id]    = doc
 
    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [
        {**docs[doc_id], "rrf_score": scores[doc_id]}
        for doc_id in sorted_ids
    ]


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Rerank candidates using a cross-encoder model.
    Returns candidates sorted by cross-encoder relevance score.
    """
    pairs  = [[query, doc["content"]] for doc in candidates]
    scores = reranker.predict(pairs)
 
    for doc, score in zip(candidates, scores):
        doc["rerank_score"] = float(score)
 
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
 
    print(f"📊 Reranking scores:")
    for doc in reranked[:TOP_K_FINAL]:
        print(f"   [{doc['rerank_score']:+.3f}] {doc['title']}")
 
    return reranked[:TOP_K_FINAL]
# ══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════


def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base using hybrid search with reranking.
    Use this when asked about policies, SLAs, escalation procedures,
    renewal playbooks, onboarding, opportunity stages, or data privacy rules.
 
    Args:
        query: The question or topic to search for.
    """
    try:
        # Step 1 — Rewrite the query for better retrieval
        search_query = rewrite_query(query)

        if RAG_LITE:
            # ponytail: 8 docs — BM25 alone retrieves fine; dense search +
            # reranking come back on hosts with enough RAM for torch
            results = _bm25_search(search_query, top_k=TOP_K_FINAL)
            if not results:
                return "No documents found in the knowledge base."
            output = f"Knowledge base results for '{query}':\n\n"
            for i, doc in enumerate(results, 1):
                output += f"[{i}] {doc['title']} (relevance: {doc['score']:+.3f})\n"
                output += f"{doc['content'].strip()}\n\n"
            return output.strip()
 
        # Step 2 — Dense search (semantic)
        dense_results = _dense_search(search_query, top_k=TOP_K_RETRIEVAL)
 
        # Step 3 — BM25 search (keyword)
        bm25_results  = _bm25_search(search_query, top_k=TOP_K_RETRIEVAL)
 
        # Step 4 — Fuse rankings with RRF
        fused = _reciprocal_rank_fusion(dense_results, bm25_results)
 
        if not fused:
            return "No documents found in the knowledge base."
 
        # Step 5 — Cross-encoder rerank the fused top-N (see RERANK_CANDIDATES)
        reranked = _rerank(query, fused[:RERANK_CANDIDATES])  # original query for reranking
 
        # Step 6 — Format output
        output = f"Knowledge base results for '{query}':\n\n"
        for i, doc in enumerate(reranked, 1):
            output += f"[{i}] {doc['title']} (relevance: {doc['rerank_score']:+.3f})\n"
            output += f"{doc['content'].strip()}\n\n"
 
        return output.strip()
 
    except Exception as e:
        return f"Knowledge base search failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  python knowledge_base.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not RAG_LITE:
        setup_collection()
        seed_documents()
 
    tests = [
        "what is the escalation policy for high priority cases?",
        "what should I do 30 days before a renewal?",
        "what does Closed Won mean?",              # previously scored 0.20 — the weak case
        "how do I create a good support case?",
        "what are the SLA response times?",
    ]
 
    for query in tests:
        print(f"\n{'═'*60}")
        print(f"QUERY: {query}")
        print('═'*60)
        result = search_knowledge_base(query)
        # Print just the titles and scores, not full content
        for line in result.split('\n'):
            if line.startswith('[') or line.startswith('Knowledge'):
                print(line)
