"""Run the same questions through every rung and compare answers, model calls, tokens and time.

    python compare.py
"""
import time
import llm
from step1_llm import llm_answer
from step2_rag import rag_answer
from step3_agent import run_agent
from step4_agentic import run_team

QUESTIONS = [
    "Can a customer return fresh bread two days after buying it?",
    "How much stock value of SKU 1001 does Kitwe hold?",
]

RUNGS = [
    ("1. LLM", llm_answer),
    ("2. RAG", rag_answer),
    ("3. Agent", run_agent),
    ("4. Agentic", run_team),
]


def main():
    rows = []
    for question in QUESTIONS:
        print(f"\n=== {question}")
        for name, fn in RUNGS:
            llm.reset_stats()
            start = time.perf_counter()
            try:
                answer = fn(question)
            except Exception as e:  # keep going so one failure doesn't hide the rest
                answer = f"ERROR: {e}"
            seconds = time.perf_counter() - start
            tokens = llm.STATS["prompt_tokens"] + llm.STATS["completion_tokens"]
            rows.append((question[:40], name, llm.STATS["calls"], tokens, seconds))
            print(f"\n--- {name}\n{(answer or '').strip()[:500]}")

    print("\n\n| Question | Rung | Model calls | Tokens | Seconds |")
    print("|---|---|---|---|---|")
    for q, name, calls, tokens, seconds in rows:
        print(f"| {q} | {name} | {calls} | {tokens} | {seconds:.1f} |")


if __name__ == "__main__":
    main()
