"""Rung 2: RAG. Retrieve the relevant policy text first, then answer from it."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm import chat


def load_chunks(folder="docs"):
    """Split every document into paragraphs. Each paragraph is one 'chunk'."""
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        for para in path.read_text().split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                chunks.append({"source": path.name, "text": para})
    return chunks


class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in chunks)

    def search(self, query, k=3, min_score=0.05):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)}
                for i in best if scores[i] >= min_score]


RETRIEVER = Retriever(load_chunks())


def rag_answer(question):
    hits = RETRIEVER.search(question)
    if not hits:
        return "I couldn't find this in the policy documents."
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    reply = chat([
        {"role": "system", "content":
            "Answer ONLY from the context below. Cite the source file in brackets. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"CONTEXT:\n{context}"},
        {"role": "user", "content": question},
    ])
    return reply.content


if __name__ == "__main__":
    print(rag_answer("Can a customer return fresh bread two days after buying it?"))
