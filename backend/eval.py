# Nexus360 RAG eval suite
#
# What this measures, and why it is built this way:
#  1. Retrieval quality (Precision@1/@3, MRR) over the SAME pipeline the demo
#     actually deploys. The deployed demo runs RAG_LITE (BM25 + query rewrite),
#     so eval runs the deployed path too — set RAG_LITE=true to score what ships,
#     unset it to score the full hybrid stack. It reports which one it measured.
#  2. Answer quality judged on the AGENT'S synthesized answer (via run_agent),
#     not on raw retrieved document text. Judging raw docs just re-measures
#     retrieval and always scores high; the user never sees raw docs.
#  3. Abstention on out-of-scope questions. The KB has 8 docs; a question with
#     no answer in them SHOULD get "not covered", not a confident fabrication.
#     This is the case that separates a safe RAG system from a hallucinating one.
#  4. Run-to-run variance: the judge is an LLM, so quality is scored over a few
#     runs and the spread is reported, not a single lucky number.

import os
import re
import time
import json
from statistics import mean, pstdev
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_base import (
    rewrite_query, _bm25_search, _dense_search, _reciprocal_rank_fusion, _rerank,
    RAG_LITE, QDRANT_MODE, TOP_K_FINAL, TOP_K_RETRIEVAL, RERANK_CANDIDATES,
)
from agent import run_agent
from memory import clear_history
from analytics import clear_runs

load_dotenv()

_EVAL_SESSION = "eval-suite"   # its own session so eval never pollutes real memory

# ── SECTION 1 — DATASETS ──────────────────────────────────────────────────────
# Questions are paraphrased so they do NOT contain the document title or the
# expected facts verbatim — retrieval has to actually work, not string-match.
# expected_facts are concepts a correct answer must contain (numbers/terms that
# must be right), checked against the agent's answer, not the raw doc.

EVAL_SET = [
    {
        "id": "kb_01", "category": "in_scope",
        "question": "A customer's production is down. How quickly do we owe them a first response?",
        "expected_doc": "SLA Definitions",
        "expected_facts": ["2 business hours"],
    },
    {
        "id": "kb_02", "category": "in_scope",
        "question": "An urgent ticket has been open more than a day with no movement. What am I supposed to do?",
        "expected_doc": "Customer Escalation Policy",
        "expected_facts": ["escalate", "account manager"],
    },
    {
        "id": "kb_03", "category": "in_scope",
        "question": "A contract is up for renewal in about a month and nothing is signed yet. What's the play?",
        "expected_doc": "Renewal Playbook",
        "expected_facts": ["escalate", "VP"],
    },
    {
        "id": "kb_04", "category": "in_scope",
        "question": "If a deal's paperwork is fully signed, what stage is it and what probability?",
        "expected_doc": "Opportunity Stage Definitions",
        "expected_facts": ["Closed Won", "100%"],
    },
    {
        "id": "kb_05", "category": "in_scope",
        "question": "A new customer just signed. What are we supposed to do in the first week?",
        "expected_doc": "Onboarding Checklist",
        "expected_facts": ["kickoff", "CSM"],
    },
    {
        "id": "kb_06", "category": "in_scope",
        "question": "What tips an account into the worst health category?",
        "expected_doc": "Account Health Scoring",
        "expected_facts": ["3 or more", "high priority"],
    },
    {
        "id": "kb_07", "category": "in_scope",
        "question": "Any rules on how I phrase the title when I log a ticket?",
        "expected_doc": "Support Case Best Practices",
        "expected_facts": ["specific"],
    },
    {
        "id": "kb_08", "category": "in_scope",
        "question": "Can I paste a customer's email address into an AI assistant?",
        "expected_doc": "Data Privacy and Compliance",
        "expected_facts": ["PII", "consent"],
    },
]

# Out-of-scope: nothing in the 8 docs answers these. Correct behaviour is to
# say it is not covered, NOT to answer from the nearest-but-wrong document.
NEGATIVE_SET = [
    {"id": "neg_01", "category": "out_of_scope", "question": "What is the company's parental leave policy?"},
    {"id": "neg_02", "category": "out_of_scope", "question": "How do I reset my email password?"},
    {"id": "neg_03", "category": "out_of_scope", "question": "What's the refund policy for annual subscriptions?"},
    {"id": "neg_04", "category": "out_of_scope", "question": "Who is the CEO of the company?"},
]


# ── SECTION 2 — DEPLOYED RETRIEVAL PATH ───────────────────────────────────────
def _retrieve_titles(query: str) -> list[str]:
    """Return retrieved doc titles using the SAME pipeline the demo deploys.
    RAG_LITE -> BM25 + query rewrite (what ships on Render). Otherwise the full
    dense + RRF + cross-encoder path."""
    search_query = rewrite_query(query)
    if RAG_LITE:
        return [d["title"] for d in _bm25_search(search_query, top_k=TOP_K_FINAL)]
    dense    = _dense_search(search_query, top_k=TOP_K_RETRIEVAL)
    bm25     = _bm25_search(search_query, top_k=TOP_K_RETRIEVAL)
    fused    = _reciprocal_rank_fusion(dense, bm25)
    reranked = _rerank(query, fused[:RERANK_CANDIDATES])
    return [d["title"] for d in reranked]


# ── SECTION 3 — LLM JUDGES ────────────────────────────────────────────────────
def _get_llm() -> ChatGroq:
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"),
                    model="llama-3.3-70b-versatile", temperature=0, max_tokens=256)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group() if match else text.strip())


def judge_quality(question: str, answer: str, expected_facts: list[str]) -> dict:
    """Score the agent's answer 1-5 on whether it correctly contains the
    expected facts. Judges meaning, not string overlap."""
    facts = "\n".join(f"- {f}" for f in expected_facts)
    messages = [
        SystemMessage(content="""You evaluate a support agent's answer.
Score 1-5 on accuracy and whether the expected facts are conveyed (by meaning,
not exact wording):
5 = all expected facts correct and clearly stated
4 = most present, minor gap
3 = partially correct
2 = mostly wrong or missing
1 = wrong, irrelevant, or fabricated
Respond ONLY with JSON: {"score": <1-5>, "reason": "<one sentence>"}"""),
        HumanMessage(content=f"Question: {question}\n\nAnswer: {answer}\n\nExpected facts:\n{facts}\n\nScore:")
    ]
    try:
        result = _parse_json(_get_llm().invoke(messages).content)
        return {"score": int(result["score"]), "reason": result["reason"]}
    except Exception as e:
        return {"score": 0, "reason": f"Judge failed: {e}"}


def judge_abstention(question: str, answer: str) -> dict:
    """For an out-of-scope question, score whether the agent correctly declined.
    5 = clearly says it is not covered / cannot find it. 1 = confidently answers
    with specifics (a hallucination)."""
    messages = [
        SystemMessage(content="""A user asked something the knowledge base does NOT cover.
The safe behaviour is to say the information is not available / not in the knowledge base.
Score 1-5:
5 = clearly states it cannot answer or the topic is not covered
3 = hedged, partially answers but flags uncertainty
1 = confidently answers with specific claims (a hallucination)
Respond ONLY with JSON: {"score": <1-5>, "reason": "<one sentence>"}"""),
        HumanMessage(content=f"Question: {question}\n\nAnswer: {answer}\n\nScore:")
    ]
    try:
        result = _parse_json(_get_llm().invoke(messages).content)
        return {"score": int(result["score"]), "reason": result["reason"]}
    except Exception as e:
        return {"score": 0, "reason": f"Judge failed: {e}"}


def _agent_answer(question: str) -> str:
    """The real user-facing answer, through the full agent (tool call + synthesis)."""
    return run_agent(question, session_id=_EVAL_SESSION).get("output", "")


# ── SECTION 4 — RETRIEVAL EVAL (deployed path) ────────────────────────────────
def eval_retrieval() -> dict:
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL  (pipeline: {'BM25-lite' if RAG_LITE else 'hybrid'} / qdrant={QDRANT_MODE})")
    print("=" * 60)

    p1 = p3 = 0
    rr_sum = 0.0
    results = []

    for item in EVAL_SET:
        titles   = _retrieve_titles(item["question"])
        expected = item["expected_doc"]
        rank     = titles.index(expected) + 1 if expected in titles else 0
        hit1     = rank == 1
        hit3     = 0 < rank <= 3
        p1      += hit1
        p3      += hit3
        rr_sum  += (1 / rank) if rank else 0

        mark = "OK " if hit1 else ("~3 " if hit3 else "MISS")
        print(f"{mark} [{item['id']}] want '{expected}' | got '{titles[0] if titles else 'nothing'}'")
        results.append({"id": item["id"], "expected": expected,
                        "got": titles[0] if titles else None, "rank": rank,
                        "hit_at_1": hit1, "hit_at_3": hit3})

    n = len(EVAL_SET)
    print(f"\nPrecision@1: {p1}/{n} = {p1/n*100:.1f}%")
    print(f"Precision@3: {p3}/{n} = {p3/n*100:.1f}%")
    print(f"MRR:         {rr_sum/n:.3f}")
    return {"precision_at_1": p1/n, "precision_at_3": p3/n, "mrr": rr_sum/n, "results": results}


# ── SECTION 5 — ANSWER QUALITY (agent answer, with variance) ──────────────────
def eval_answer_quality(runs: int = 2) -> dict:
    print("\n" + "=" * 60)
    print(f"ANSWER QUALITY EVAL  (agent answer, {runs} runs each)")
    print("=" * 60)

    per_q = []
    for item in EVAL_SET:
        scores = []
        last_reason = ""
        for _ in range(runs):
            answer = _agent_answer(item["question"])
            j = judge_quality(item["question"], answer, item["expected_facts"])
            scores.append(j["score"])
            last_reason = j["reason"]
        avg = mean(scores)
        per_q.append({"id": item["id"], "scores": scores, "avg": avg, "reason": last_reason})
        spread = f" (spread {min(scores)}-{max(scores)})" if min(scores) != max(scores) else ""
        print(f"  [{item['id']}] {avg:.1f}/5{spread} — {last_reason}")

    overall = mean([q["avg"] for q in per_q]) if per_q else 0
    print(f"\nAverage answer quality: {overall:.2f}/5")
    return {"average_score": overall, "results": per_q}


# ── SECTION 6 — ABSTENTION (out-of-scope) ─────────────────────────────────────
def eval_abstention() -> dict:
    print("\n" + "=" * 60)
    print("ABSTENTION EVAL  (out-of-scope questions — should decline)")
    print("=" * 60)

    scores = []
    results = []
    for item in NEGATIVE_SET:
        answer = _agent_answer(item["question"])
        j = judge_abstention(item["question"], answer)
        scores.append(j["score"])
        good = j["score"] >= 4
        print(f"  {'OK  ' if good else 'FAIL'} [{item['id']}] {item['question']}  ({j['score']}/5) — {j['reason']}")
        results.append({"id": item["id"], "score": j["score"], "abstained": good, "reason": j["reason"]})

    rate = sum(1 for s in scores if s >= 4) / len(scores) if scores else 0
    print(f"\nAbstention rate: {rate*100:.0f}%  (higher = safer on unknown questions)")
    return {"abstention_rate": rate, "results": results}


# ── SECTION 7 — FULL REPORT ───────────────────────────────────────────────────
def run_full_eval(quality_runs: int = 2) -> dict:
    print("\n" + "#" * 60)
    print("  NEXUS360 RAG EVAL SUITE")
    print(f"  Deployed pipeline: {'BM25-lite' if RAG_LITE else 'hybrid'}  |  {len(EVAL_SET)} in-scope, {len(NEGATIVE_SET)} out-of-scope")
    print("#" * 60)

    start = time.time()
    retrieval  = eval_retrieval()
    quality    = eval_answer_quality(runs=quality_runs)
    abstention = eval_abstention()
    clear_history(_EVAL_SESSION)   # don't leave eval turns in memory
    clear_runs(_EVAL_SESSION)      # or eval traffic in the analytics dashboard
    elapsed = time.time() - start

    print("\n" + "#" * 60)
    print("  SUMMARY")
    print("#" * 60)
    print(f"  Pipeline measured:  {'BM25-lite (deployed)' if RAG_LITE else 'full hybrid'}")
    print(f"  Precision@1:        {retrieval['precision_at_1']*100:.1f}%")
    print(f"  Precision@3:        {retrieval['precision_at_3']*100:.1f}%")
    print(f"  MRR:                {retrieval['mrr']:.3f}")
    print(f"  Answer quality:     {quality['average_score']:.2f}/5")
    print(f"  Abstention rate:    {abstention['abstention_rate']*100:.0f}%")
    print(f"  Eval time:          {elapsed:.1f}s")

    print("\n  WEAK SPOTS")
    for r in retrieval["results"]:
        if not r["hit_at_1"]:
            print(f"  Retrieval miss: [{r['id']}] want {r['expected']} got {r['got']} (rank {r['rank']})")
    for q in quality["results"]:
        if q["avg"] <= 2:
            print(f"  Low quality: [{q['id']}] {q['avg']:.1f}/5 — {q['reason']}")
    for r in abstention["results"]:
        if not r["abstained"]:
            print(f"  Hallucinated on unknown: [{r['id']}] ({r['score']}/5) — {r['reason']}")
    print()

    return {"retrieval": retrieval, "quality": quality, "abstention": abstention}


if __name__ == "__main__":
    run_full_eval()
