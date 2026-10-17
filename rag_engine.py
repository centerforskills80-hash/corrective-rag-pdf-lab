"""
Corrective RAG (CRAG) engine for a student lab.
- Reads text from multiple PDFs
- Splits text into overlapping chunks
- Uses TF-IDF retrieval (no paid embedding API)
- Grades retrieved results and retries with broader retrieval when weak
- Generates answers with local Ollama OR Groq API (free tier; limits apply)
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import re

import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    text: str
    source: str
    page: int


class CorrectiveRAG:
    def __init__(self, chunk_size: int = 900, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def add_pdfs(self, uploaded_files) -> int:
        """Read uploaded PDF file objects and add their text to the knowledge base."""
        added = 0
        for pdf_file in uploaded_files:
            try:
                reader = PdfReader(pdf_file)
                filename = getattr(pdf_file, "name", "uploaded.pdf")
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    text = re.sub(r"\s+", " ", text).strip()
                    if not text:
                        continue
                    for part in self._split_text(text):
                        self.chunks.append(Chunk(part, filename, page_num))
                        added += 1
            except Exception as exc:
                raise ValueError(f"Could not read {getattr(pdf_file, 'name', 'a PDF')}: {exc}") from exc

        if added:
            self._rebuild_index()
        return added

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        parts = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            part = text[start:end].strip()
            if part:
                parts.append(part)
            if end == len(text):
                break
            start = max(end - self.overlap, start + 1)
        return parts

    def _rebuild_index(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, question: str, top_k: int = 4) -> Dict:
        if not self.chunks or self.vectorizer is None:
            return {"chunks": [], "score": 0.0, "corrected": False, "message": "Upload and index PDFs first."}

        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        first_ids = [int(i) for i in ranked[:top_k]]
        first_score = float(scores[first_ids[0]]) if first_ids else 0.0
        corrected = False

        # CRAG-style correction: if retrieval confidence is weak, broaden the candidate set.
        # TF-IDF scores are not probabilities; this is a simple teaching heuristic.
        if first_score < 0.08:
            corrected = True
            expanded_k = min(max(top_k * 2, 8), len(self.chunks))
            ids = [int(i) for i in ranked[:expanded_k]]
            # Keep only non-zero matches if possible; otherwise return the best available chunks.
            nonzero_ids = [i for i in ids if scores[i] > 0]
            if nonzero_ids:
                ids = nonzero_ids[:top_k]
            else:
                ids = ids[:top_k]
        else:
            ids = first_ids

        results = []
        for i in ids:
            results.append({
                "text": self.chunks[i].text,
                "source": self.chunks[i].source,
                "page": self.chunks[i].page,
                "score": float(scores[i]),
            })
        best_score = max((r["score"] for r in results), default=0.0)
        return {
            "chunks": results,
            "score": best_score,
            "corrected": corrected,
            "message": "Expanded retrieval used because the first match was weak." if corrected else "Initial retrieval looked relevant."
        }

    def answer(self, question: str, provider: str = "Ollama",
               model: str = "llama3.2:3b", api_key: str = "",
               top_k: int = 4) -> Dict:
        retrieved = self.retrieve(question, top_k=top_k)
        if not retrieved["chunks"] or retrieved["score"] <= 0:
            return {
                "answer": "I could not find relevant text in the uploaded PDFs. Try rephrasing the question or uploading a document that covers the topic.",
                "retrieval": retrieved
            }

        context = "\n\n".join(
            f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
            for c in retrieved["chunks"]
        )
        prompt = f"""You are a careful document question-answering assistant.
Answer using ONLY the context below. If the context does not contain the answer, say so clearly.
Do not invent facts. Cite supporting sources using [filename, page number].

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        try:
            if provider == "Ollama":
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=180
                )
                response.raise_for_status()
                answer_text = response.json().get("response", "").strip()
            elif provider == "Groq API":
                if not api_key:
                    raise ValueError("Enter a Groq API key in the sidebar.")
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "Answer only from the provided context. If information is missing, say so. Cite sources."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    },
                    timeout=120
                )
                response.raise_for_status()
                answer_text = response.json()["choices"][0]["message"]["content"].strip()
            else:
                raise ValueError("Choose Ollama or Groq API.")
            return {"answer": answer_text, "retrieval": retrieved}
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to the selected model service. For Ollama, install/start Ollama and pull the selected model. "
                "For Groq, check your internet connection and API key."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = response.text[:500]
            except Exception:
                pass
            raise RuntimeError(f"Model service returned an error: {detail or exc}") from exc
