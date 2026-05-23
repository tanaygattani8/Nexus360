# ── Imports ───────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
COLLECTION_NAME = "nexus360_knowledge"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # small, fast, 384 dimensions
VECTOR_SIZE      = 384


# ── Clients ───────────────────────────────────────────────────────────────────
# Both created once at module level and reused

def _get_qdrant() -> QdrantClient:
    url     = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise EnvironmentError("QDRANT_URL or QDRANT_API_KEY not set in .env")
    return QdrantClient(url=url, api_key=api_key)

# SentenceTransformer downloads the model on first run (~90MB, cached after)
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Embedding model loaded")


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
# These are fake but realistic internal docs — playbooks, SLAs, policies.
# In a real company these would be pulled from Confluence, Notion, or Google Drive.
# Each doc has a title, content, and category for filtering.

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
        Qualification: Budget, authority, need, and timeline confirmed (BANT). Probability: 20%
        Needs Analysis: Deep discovery completed, pain points documented. Probability: 30%
        Value Proposition: Custom demo or POC delivered. Probability: 40%
        Id. Decision Makers: All stakeholders identified and engaged. Probability: 60%
        Perception Analysis: Objections handled, competitive positioning done. Probability: 70%
        Proposal/Price Quote: Formal proposal sent and reviewed. Probability: 75%
        Negotiation/Review: Contract in legal review or active negotiation. Probability: 90%
        Closed Won: Contract signed. Probability: 100%
        Closed Lost: Deal lost. Always fill in the Loss Reason field.
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
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════
# The function the agent calls at query time.
# Embeds the query, finds the top-k most similar documents, returns them.

def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    Search the internal knowledge base for documents relevant to the query.
    Use this when asked about policies, playbooks, SLAs, or internal processes.

    Args:
        query: The question or topic to search for.
        top_k: Number of results to return (default 3).

    Returns:
        Formatted string of the most relevant document excerpts.
    """
    try:
        client  = _get_qdrant()
        vector  = embedder.encode(query).tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
        ).points

        if not results:
            return "No relevant documents found in the knowledge base."

        output = f"Knowledge base results for '{query}':\n\n"
        for i, hit in enumerate(results, 1):
            output += f"[{i}] {hit.payload['title']} (score: {hit.score:.2f})\n"
            output += f"{hit.payload['content'].strip()}\n\n"

        return output.strip()

    except Exception as e:
        return f"Knowledge base search failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  python knowledge_base.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── Setting up collection ────────────────────────────")
    setup_collection()

    print("\n── Seeding documents ────────────────────────────────")
    seed_documents()

    print("\n── Test search 1: escalation policy ─────────────────")
    result = search_knowledge_base("what is the escalation policy for high priority cases?")
    print(result[:500] + "...")

    print("\n── Test search 2: renewal process ───────────────────")
    result = search_knowledge_base("what should I do 30 days before a renewal?")
    print(result[:500] + "...")

    print("\n── Test search 3: opportunity stages ────────────────")
    result = search_knowledge_base("what does Closed Won mean?")
    print(result[:500] + "...")
