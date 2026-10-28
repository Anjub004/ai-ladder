"""Rung 3: an AI agent. The model decides which tools to call, in a loop."""
import ast, csv, json, operator
from llm import chat
from step2_rag import RETRIEVER


# ---- Tools: plain Python functions the model is allowed to use ----
def search_policies(query: str) -> str:
    return json.dumps(RETRIEVER.search(query))


def check_stock(sku: str, branch: str) -> str:
    with open("stock.csv") as f:
        for row in csv.DictReader(f):
            if row["sku"] == sku and row["branch"].lower() == branch.lower():
                return json.dumps(row)
    return json.dumps({"error": f"No record for SKU {sku} at {branch}"})


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

def calculate(expression: str) -> str:
    """Safe arithmetic only (+ - * /). Never use eval() on model output."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Only numbers and + - * / are allowed")
    return str(walk(ast.parse(expression, mode="eval").body))


TOOLS = {"search_policies": search_policies, "check_stock": check_stock, "calculate": calculate}

def schema(name, description, **params):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "required": list(params),
                       "properties": {p: {"type": "string", "description": d} for p, d in params.items()}}}}

TOOL_SCHEMAS = [
    schema("search_policies", "Search store policy and SOP documents.", query="What to look for"),
    schema("check_stock", "Get quantity and unit price for a SKU at a branch.", sku="SKU code, e.g. 1001", branch="Branch name"),
    schema("calculate", "Do arithmetic.", expression="e.g. 12 * 32.5"),
]


def run_agent(task, system="You are a retail operations assistant. Use tools; never guess numbers.", max_steps=6):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for step in range(max_steps):
        reply = chat(messages, tools=TOOL_SCHEMAS)
        messages.append(reply.model_dump(exclude_none=True))
        if not reply.tool_calls:              # no tool requested -> final answer
            return reply.content
        for call in reply.tool_calls:         # run each requested tool, feed results back
            name, args = call.function.name, {}
            try:
                args = json.loads(call.function.arguments or "{}")
                if name not in TOOLS:
                    raise ValueError(f"Unknown tool: {name}")
                result = TOOLS[name](**args)
            except Exception as e:            # errors go back to the model, not up to the user
                result = json.dumps({"error": str(e)})
            print(f"  [step {step + 1}] {name}({args}) -> {result[:80]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return "Stopped: step limit reached without a final answer."


if __name__ == "__main__":
    print(run_agent("How much stock value of SKU 1001 does Kitwe hold, and what is the policy if some of it arrived damaged?"))
