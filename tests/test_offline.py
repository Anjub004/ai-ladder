"""Offline tests: a scripted fake model stands in for the LLM, so no server or API key is needed.

    pytest -q
"""
import json
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import llm  # noqa: E402
from openai.types.chat import ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion_message_tool_call import (  # noqa: E402
    ChatCompletionMessageToolCall, Function)


class FakeCompletions:
    """Returns scripted replies in order. A reply can be a message or a function(messages, kwargs)."""
    def __init__(self):
        self.script, self.seen = [], []

    def create(self, model, messages, **kwargs):
        self.seen.append((messages, kwargs))
        item = self.script.pop(0)
        msg = item(messages, kwargs) if callable(item) else item
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


@pytest.fixture
def fake(monkeypatch):
    completions = FakeCompletions()
    monkeypatch.setattr(llm, "client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    llm.reset_stats()
    return completions


def text(content):
    return ChatCompletionMessage(role="assistant", content=content)


def tool_call(i, name, args):
    return ChatCompletionMessageToolCall(id=f"c{i}", type="function",
                                         function=Function(name=name, arguments=json.dumps(args)))


def test_retriever_finds_policy_and_refuses_nonsense():
    from step2_rag import RETRIEVER
    hits = RETRIEVER.search("return fresh bread two days after buying")
    assert any("Perishable" in h["text"] for h in hits)
    assert RETRIEVER.search("quantum chromodynamics") == []


def test_rag_puts_policy_in_prompt(fake):
    from step2_rag import rag_answer
    fake.script.append(lambda m, k: text("OK" if "returns_policy.md" in m[0]["content"] else "MISSING"))
    assert rag_answer("Can a customer return fresh bread two days after buying it?") == "OK"


def test_rag_refuses_without_calling_model(fake):
    from step2_rag import rag_answer
    assert "couldn't find" in rag_answer("quantum chromodynamics")
    assert llm.STATS["calls"] == 0


def test_calculator_is_safe():
    from step3_agent import calculate
    assert calculate("12 * 32.5") == "390.0"
    with pytest.raises(ValueError):
        calculate("__import__('os').system('echo hi')")


def test_agent_runs_tools_and_handles_errors(fake):
    from step3_agent import run_agent
    fake.script += [
        ChatCompletionMessage(role="assistant", content=None, tool_calls=[
            tool_call(1, "check_stock", {"sku": "1001", "branch": "kitwe"}),
            tool_call(2, "nonexistent", {})]),
        lambda m, k: text("FINAL " + " | ".join(x["content"] for x in m if x["role"] == "tool")),
    ]
    answer = run_agent("test")
    assert '"qty": "12"' in answer
    assert "Unknown tool: nonexistent" in answer
    assert llm.STATS["calls"] == 2


def test_agent_stops_at_step_limit(fake):
    from step3_agent import run_agent
    loop = ChatCompletionMessage(role="assistant", content=None,
                                 tool_calls=[tool_call(1, "calculate", {"expression": "1+1"})])
    fake.script += [loop] * 3
    assert "step limit" in run_agent("loop forever", max_steps=3)


def test_team_filters_unknown_agents(fake):
    from step4_agentic import run_team
    fake.script += [
        text(json.dumps({"subtasks": [{"agent": "stock", "task": "bread at Kitwe"},
                                      {"agent": "hacker", "task": "steal"}]})),
        text("Kitwe has 0 bread"),
        lambda m, k: text(f"REVIEWED {m[1]['content'].count(chr(34) + 'agent' + chr(34))} findings"),
    ]
    assert run_team("goal") == "REVIEWED 1 findings"
