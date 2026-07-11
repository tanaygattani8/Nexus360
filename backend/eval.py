# Imports
import os
import re
import time
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from knowledge_base import search_knowledge_base, _dense_search, _bm25_search, _rerank, rewrite_query, DOCUMENTS

load_dotenv()

# SECTION 1 - EVAL DATASET
EVAL_SET = [
    {
        "id":               "kb_01",
        "category":         "knowledge_base",
        "question":         "What is the SLA for high priority cases?",
        "expected_doc":     "SLA Definitions",
        "expected_facts":   ["2 business hours", "24 business hours", "first response"],
    },
    {
        "id":            "kb_02",
        "category":      "knowledge_base",
        "question":      "When should I escalate a support case?",
        "expected_doc":  "Customer Escalation Policy",
        "expected_facts": ["24 hours", "account manager", "data loss"],
    },
    {
        "id":            "kb_03",
        "category":      "knowledge_base",
        "question":      "What should I do 30 days before a renewal?",
        "expected_doc":  "Renewal Playbook",
        "expected_facts": ["30 days", "contract", "VP"],
    },
    {
        "id":            "kb_04",
        "category":      "knowledge_base",
        "question":      "What does Closed Won mean in Salesforce?",
        "expected_doc":  "Opportunity Stage Definitions",
        "expected_facts": ["contract signed", "100%", "probability"],
    },
    {
        "id":            "kb_05",
        "category":      "knowledge_base",
        "question":      "What are the steps to onboard a new customer?",
        "expected_doc":  "Onboarding Checklist",
        "expected_facts": ["kickoff", "CSM", "week 1"],
    },
    {
        "id":            "kb_06",
        "category":      "knowledge_base",
        "question":      "What makes an account Red in health scoring?",
        "expected_doc":  "Account Health Scoring",
        "expected_facts": ["3 or more", "high priority", "90 days"],
    },
    {
        "id":            "kb_07",
        "category":      "knowledge_base",
        "question":      "How should I write a support case subject line?",
        "expected_doc":  "Support Case Best Practices",
        "expected_facts": ["specific", "subject"],
    },

    # Edge Cases
    {
        "id":            "edge_01",
        "category":      "edge",
        "question":      "what happens with closed won",   # lowercase, no punctuation
        "expected_doc":  "Opportunity Stage Definitions",
        "expected_facts": ["contract", "100"],
    },
    {
        "id":            "edge_02",
        "category":      "edge",
        "question":      "SLA breach high priority",       # keyword-style query
        "expected_doc":  "SLA Definitions",
        "expected_facts": ["2 business hours", "VP"],
    },
]


# SECTION 2 - LLM JUDGE
def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=256,
    )

def llm_judge(question: str, answer: str, expected_facts: list[str]) -> dict:
    """
    Score an answer 1-5 based on whether it contains the expected facts.
 
    Returns:
        { "score": int, "reason": str }
    """
    llm = _get_llm()
    facts_str = "\n".join(f"- {f}" for f in expected_facts)

    messages = [
        SystemMessage(content="""You are an answer quality evaluator.
Score the answer on a scale of 1-5 based on how well it addresses the question
and contains the expected key facts.
 
Scoring guide:
5 = All expected facts present, answer is accurate and clear
4 = Most expected facts present, minor gaps
3 = Some expected facts present, partially correct
2 = Few expected facts present, mostly incorrect or incomplete
1 = Expected facts missing, answer is wrong or irrelevant
 
Respond ONLY with valid JSON in this exact format:
{"score": <1-5>, "reason": "<one sentence explanation>"}"""),
        HumanMessage(content=f"""Question: {question}
 
Answer: {answer}
 
Expected facts that should appear in the answer:
{facts_str}
 
Score this answer:""")
    ]

    try:
        response = llm.invoke(messages)
        # Extract the JSON object even if the model wraps it in ```json fences
        match  = re.search(r"\{.*\}", response.content, re.DOTALL)
        result = json.loads(match.group() if match else response.content.strip())
        return {"score": int(result["score"]), "reason": result["reason"]}
    except Exception as e:
        return {"score": 0, "reason": f"Judge failed: {e}"}


# SECTION 3 - RETRIEVAL EVAL
def eval_retrieval(eval_set: list[dict]) -> dict:
    """
    Run retrieval eval on the eval set.
    Returns precision@1 and precision@3 scores.
    """
    print("\n" + "═"*60)
    print("RETRIEVAL EVAL")
    print("═"*60)
 
    p_at_1 = 0
    p_at_3 = 0
    results = []
 
    for item in eval_set:
        q            = item["question"]
        expected_doc = item["expected_doc"]
 
        # Run the full hybrid pipeline
        rewritten     = rewrite_query(q)
        dense_results = _dense_search(rewritten, top_k=6)
        bm25_results  = _bm25_search(rewritten, top_k=6)
 
        # RRF fusion
        from knowledge_base import _reciprocal_rank_fusion
        fused    = _reciprocal_rank_fusion(dense_results, bm25_results)
        reranked = _rerank(q, fused)
 
        retrieved_titles = [doc["title"] for doc in reranked]
 
        hit_at_1 = retrieved_titles[0] == expected_doc if retrieved_titles else False
        hit_at_3 = expected_doc in retrieved_titles[:3]
 
        if hit_at_1: p_at_1 += 1
        if hit_at_3: p_at_3 += 1
 
        status = "✅" if hit_at_1 else ("⚠️ " if hit_at_3 else "❌")
        print(f"{status} [{item['id']}] Expected: '{expected_doc}' | Got: '{retrieved_titles[0] if retrieved_titles else 'nothing'}'")
 
        results.append({
            "id":          item["id"],
            "question":    q,
            "expected":    expected_doc,
            "got":         retrieved_titles[0] if retrieved_titles else None,
            "hit_at_1":    hit_at_1,
            "hit_at_3":    hit_at_3,
            "top_3":       retrieved_titles[:3],
        })
 
    total = len(eval_set)
    print(f"\n📊 Precision@1: {p_at_1}/{total} = {p_at_1/total*100:.1f}%")
    print(f"📊 Precision@3: {p_at_3}/{total} = {p_at_3/total*100:.1f}%")
 
    return {
        "precision_at_1": p_at_1 / total,
        "precision_at_3": p_at_3 / total,
        "results":        results,
    }

# SECTION 4 - ANSWER QUALITY EVAL
def eval_answer_quality(eval_set: list[dict]) -> dict:
    """
    Run answer quality eval on the eval set.
    Returns average score and per-question results.
    """
    print("\n" + "═"*60)
    print("ANSWER QUALITY EVAL")
    print("═"*60)
 
    scores  = []
    results = []
 
    for item in eval_set:
        q = item["question"]
        print(f"\n[{item['id']}] {q}")
 
        # Get the full answer from the pipeline
        answer = search_knowledge_base(q)
 
        # Judge the answer
        judgement = llm_judge(q, answer, item["expected_facts"])
        score     = judgement["score"]
        scores.append(score)
 
        stars = "★" * score + "☆" * (5 - score)
        print(f"  Score: {stars} ({score}/5) — {judgement['reason']}")
 
        results.append({
            "id":       item["id"],
            "question": q,
            "score":    score,
            "reason":   judgement["reason"],
        })
 
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n📊 Average answer quality: {avg:.2f}/5")
    print(f"📊 Score distribution: {sorted(scores)}")
 
    return {
        "average_score": avg,
        "results":       results,
    }


# LATENCY EVAL
def eval_latency(eval_set: list[dict], runs: int = 2) -> dict:
    """
    Measure end-to-end search latency.
    """
    print("\n" + "═"*60)
    print("LATENCY EVAL")
    print("═"*60)
 
    latencies = []
 
    for item in eval_set[:5]:   # test on first 5 to keep it fast
        q = item["question"]
        run_times = []
 
        for _ in range(runs):
            start = time.time()
            search_knowledge_base(q)
            elapsed = time.time() - start
            run_times.append(elapsed)
 
        avg_time = sum(run_times) / len(run_times)
        latencies.append(avg_time)
        print(f"  [{item['id']}] avg {avg_time:.2f}s")
 
    mean_latency = sum(latencies) / len(latencies)
    max_latency  = max(latencies)
    min_latency  = min(latencies)
 
    print(f"\n📊 Mean latency:  {mean_latency:.2f}s")
    print(f"📊 Max latency:   {max_latency:.2f}s")
    print(f"📊 Min latency:   {min_latency:.2f}s")
 
    return {
        "mean_latency": mean_latency,
        "max_latency":  max_latency,
        "min_latency":  min_latency,
    }

# FULL EVALUATION REPORT
def run_full_eval(run_latency: bool = False):
    """
    Run the full eval suite and print a summary report.
 
    Args:
        run_latency: Set True to include latency eval (adds ~30s extra).
    """
    print("\n" + "█"*60)
    print("  NEXUS360 RAG EVAL SUITE")
    print("█"*60)
    print(f"  Eval set size: {len(EVAL_SET)} questions")
    print(f"  Categories: {set(e['category'] for e in EVAL_SET)}")
 
    start_time = time.time()
 
    # Run evals
    retrieval_results = eval_retrieval(EVAL_SET)
    quality_results   = eval_answer_quality(EVAL_SET)
 
    latency_results = None
    if run_latency:
        latency_results = eval_latency(EVAL_SET)
 
    total_time = time.time() - start_time
 
    # ── Summary report ─────────────────────────────────────────────────────────
    print("\n" + "█"*60)
    print("  EVAL SUMMARY")
    print("█"*60)
    print(f"\n  RETRIEVAL")
    print(f"  Precision@1:       {retrieval_results['precision_at_1']*100:.1f}%")
    print(f"  Precision@3:       {retrieval_results['precision_at_3']*100:.1f}%")
    print(f"\n  ANSWER QUALITY")
    print(f"  Average score:     {quality_results['average_score']:.2f}/5")
 
    if latency_results:
        print(f"\n  LATENCY")
        print(f"  Mean:              {latency_results['mean_latency']:.2f}s")
        print(f"  Max:               {latency_results['max_latency']:.2f}s")
 
    print(f"\n  Total eval time:   {total_time:.1f}s")
    print("█"*60)
 
    # Flag weak spots
    print("\n  ⚠️  WEAK SPOTS")
    for r in retrieval_results["results"]:
        if not r["hit_at_1"]:
            print(f"  Retrieval miss: [{r['id']}] '{r['question']}'")
            print(f"    Expected: {r['expected']} | Got: {r['got']}")
 
    for r in quality_results["results"]:
        if r["score"] <= 2:
            print(f"  Low quality answer: [{r['id']}] score={r['score']}/5")
            print(f"    {r['reason']}")
 
    print()
    return {
        "retrieval": retrieval_results,
        "quality":   quality_results,
        "latency":   latency_results,
    }

if __name__ == "__main__":
    # Set run_latency=True to also measure pipeline speed
    run_full_eval(run_latency=False)