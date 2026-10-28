"""Rung 1: a plain LLM. It answers only from what it learned in training."""
from llm import chat


def llm_answer(question):
    reply = chat([
        {"role": "system", "content": "You are a helpful retail assistant."},
        {"role": "user", "content": question},
    ])
    return reply.content


if __name__ == "__main__":
    print(llm_answer("Can a customer return fresh bread two days after buying it?"))
