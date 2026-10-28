"""Rung 4: agentic AI. A planner splits the goal, specialist agents work, a reviewer checks."""
import json
from llm import chat
from step3_agent import run_agent

SPECIALISTS = {
    "policy": "You answer policy questions. Always use search_policies and cite the source file.",
    "stock": "You answer stock and value questions. Use check_stock and calculate; never guess numbers.",
}


def plan(goal):
    reply = chat([
        {"role": "system", "content":
            'Split the goal into at most 4 subtasks. Reply with JSON only: '
            '{"subtasks": [{"agent": "policy" or "stock", "task": "..."}]}'},
        {"role": "user", "content": goal},
    ], response_format={"type": "json_object"})
    subtasks = json.loads(reply.content)["subtasks"]
    return [s for s in subtasks if s.get("agent") in SPECIALISTS][:4]   # validate, don't trust


def review(goal, findings):
    reply = chat([
        {"role": "system", "content":
            "You are a strict reviewer. Combine the findings into one briefing for a branch manager. "
            "Flag anything unsupported, contradictory or missing. Do not add new facts."},
        {"role": "user", "content": f"GOAL: {goal}\n\nFINDINGS:\n{json.dumps(findings, indent=2)}"},
    ])
    return reply.content


def run_team(goal):
    findings = []
    for sub in plan(goal):
        print(f"-> {sub['agent']} agent: {sub['task']}")
        answer = run_agent(sub["task"], system=SPECIALISTS[sub["agent"]])
        findings.append({**sub, "answer": answer})
    return review(goal, findings)


if __name__ == "__main__":
    print(run_team("Prepare a morning briefing for Kitwe: bread and milk stock position, "
                   "and what staff must do about damaged items and promo price errors."))
