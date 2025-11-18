# Single Jupyter cell: RAG over PDFs using Google GenAI (Gemini) for embeddings & generation
# Requirements:
# pip install -U google-genai pdfplumber faiss-cpu nltk
# Set environment variable GEMINI_API_KEY (or pass it to Client below)
#
# Notes:
# - Uses google-genai SDK: client.embeddings.create(...) and client.models.generate_content(...)
# - Works with Gemini Developer API (API key). For Vertex usage, instantiate Client with vertexai=True.
# Docs: Gemini SDK & embeddings + generation (see citations after cell).

import os, re, json, textwrap
from typing import List, Dict
import pdfplumber
import faiss
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# ==== Google GenAI (Gemini) client ====
try:
    from google import genai
except Exception as e:
    raise ImportError("Install google-genai: pip install google-genai") from e

# Use environment variable GEMINI_API_KEY, or set directly here (not recommended in notebooks)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Set GEMINI_API_KEY in env (Gemini Developer API key).")

# Create client for Gemini Developer API (for Vertex use vertexai=True + project/location)
client = genai.Client(api_key=GEMINI_API_KEY)

# ==== CONFIG ====
EMBED_MODEL = "gemini-embedding-001"   # embedding model name (adjust if different)
GEN_MODEL   = "gemini-2.5-flash"       # text generation model (adjust per availability)
TOP_K = 5
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 60

# ==== PDF -> pages extractor ====
def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            txt = re.sub(r"\s+", " ", txt).strip()
            if txt:
                pages.append({"doc_id": os.path.basename(pdf_path), "page": i+1, "text": txt})
    return pages

# ==== chunking by sentence boundaries (word-count approx) ====
def chunk_text(text: str, chunk_size_words=CHUNK_SIZE_WORDS, overlap_words=CHUNK_OVERLAP_WORDS) -> List[str]:
    sents = sent_tokenize(text)
    chunks, cur, cur_words = [], "", 0
    for s in sents:
        w = len(s.split())
        if cur_words + w <= chunk_size_words or cur_words == 0:
            cur = (cur + " " + s).strip()
            cur_words += w
        else:
            chunks.append(cur.strip())
            # create overlap tail
            tail_words = cur.split()[-overlap_words:] if overlap_words > 0 else []
            cur = " ".join(tail_words + [s]).strip()
            cur_words = len(cur.split())
    if cur:
        chunks.append(cur.strip())
    return chunks

# ==== Build corpus from PDFs (pages -> chunks) ====
def build_corpus_from_pdfs(pdf_paths: List[str]) -> List[Dict]:
    corpus = []
    uid = 0
    for p in pdf_paths:
        pages = extract_pdf_pages(p)
        for pg in pages:
            chunks = chunk_text(pg["text"])
            for c in chunks:
                uid += 1
                corpus.append({"id": f"chunk_{uid}", "doc_id": pg["doc_id"], "page": pg["page"], "text": c})
    if not corpus:
        raise ValueError("No text extracted from provided PDFs.")
    return corpus

# ==== Embeddings via Gemini (batched) ====
def embed_texts_gemini(texts: List[str], model: str = EMBED_MODEL, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # resp.data is list of objects with .embedding
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings

# ==== FAISS index wrapper ====
class VectorIndex:
    def __init__(self):
        self.index = None
        self.meta = []
        self.dim = None

    def build(self, corpus: List[Dict]):
        texts = [c["text"] for c in corpus]
        embs = embed_texts_gemini(texts)
        import numpy as np
        embs_np = np.array(embs).astype('float32')
        self.dim = embs_np.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embs_np)
        self.meta = [{"id": c["id"], "doc_id": c["doc_id"], "page": c["page"], "text": c["text"]} for c in corpus]

    def query(self, q: str, top_k: int = TOP_K):
        q_emb = embed_texts_gemini([q])[0]
        import numpy as np
        q_np = np.array([q_emb]).astype('float32')
        D, I = self.index.search(q_np, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])
        return results

# ==== Safe calculator tool ====
def safe_calculator(expr: str) -> str:
    import re
    if re.fullmatch(r"[\d\.\+\-\*\/\^\(\) %]+", expr.replace(" ", "")) is None:
        return "Calculator: invalid expression"
    expr = expr.replace("^", "**")
    try:
        return str(eval(expr, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Calculator error: {e}"

# ==== Gemini LLM helpers ====
def generate_with_gemini(prompt: str, model: str = GEN_MODEL, max_output_tokens: int = 512, temperature: float = 0.0):
    # uses client.models.generate_content(...) which returns a response object with .text (or .content)
    resp = client.models.generate_content(model=model, contents=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
    # many SDK responses expose .text or .candidates[0].content
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # fallback to inspect raw fields
    try:
        return resp.candidates[0].content[0].text
    except Exception:
        return str(resp)

# ==== RAG agent orchestration ====
def rag_over_pdfs(pdf_paths: List[str]):
    print("Building corpus from PDFs...")
    corpus = build_corpus_from_pdfs(pdf_paths)
    print(f"Created {len(corpus)} chunks from {len(pdf_paths)} PDFs.")
    idx = VectorIndex()
    print("Embedding chunks and building FAISS index (this may take a while)...")
    idx.build(corpus)
    print("Index built.")
    return idx

def ask(idx: VectorIndex, question: str, top_k: int = TOP_K):
    retrieved = idx.query(question, top_k)
    prov = "\n".join([f"- {r['id']} (file:{r['doc_id']}, page:{r['page']})\n  {r['text'][:400]}..." for r in retrieved])

    # Ask Gemini to output a JSON plan first (agentic)
    system = textwrap.dedent("""
        You are an agentic assistant with access to a single tool: calculator(expression).
        First line: output a JSON object ONLY with keys:
          { "plan": [ {"action":"calculator"|"none", "input":"..."} , ... ], "final_instruction":"..." }
        After that JSON (on the first line) you may optionally include a brief rationale.
        Use the retrieved PDF chunks provided below as the only source of facts; do not hallucinate.
    """).strip()

    user_prompt = f"User question: {question}\n\nRetrieved chunks:\n{prov}\n\nProduce the JSON plan (first line) and optionally a short rationale."

    llm_resp = generate_with_gemini(system + "\n\n" + user_prompt, temperature=0.0, max_output_tokens=400)
    first_line = llm_resp.strip().split("\n")[0]
    try:
        plan = json.loads(first_line)
    except Exception:
        # retry with stricter instruction
        strict = "Respond ONLY with the JSON object described and nothing else. " + user_prompt
        llm_resp2 = generate_with_gemini(system + "\n\n" + strict, temperature=0.0, max_output_tokens=300)
        plan = json.loads(llm_resp2.strip())

    # execute plan (calculator only)
    tool_results = []
    for step in plan.get("plan", []):
        action = step.get("action", "none")
        inp = step.get("input", "")
        if action == "calculator":
            out = safe_calculator(inp)
            tool_results.append({"action": "calculator", "input": inp, "output": out})
        else:
            tool_results.append({"action": action, "input": inp, "output": None})

    # final composition: send retrieved + tools + plan to Gemini and require provenance citations
    final_system = "You are a helpful assistant. Use ONLY the retrieved PDF chunks and tool outputs. Cite file and page for any non-trivial fact."
    final_user = textwrap.dedent(f"""
        User question: {question}

        Retrieved chunks:
        {prov}

        Tool results:
        {json.dumps(tool_results, indent=2)}

        Agent plan:
        {json.dumps(plan, indent=2)}

        Instruction: {plan.get("final_instruction", "Integrate the retrieved content and tool outputs, answer concisely and cite file:page for facts.")}
    """)
    final_answer = generate_with_gemini(final_system + "\n\n" + final_user, max_output_tokens=800, temperature=0.0)

    # Print succinctly
    print("\n=== ANSWER ===\n")
    print(final_answer)
    print("\n=== PROVENANCE ===\n")
    for r in retrieved:
        print(f"{r['id']}  file:{r['doc_id']}  page:{r['page']}\n{r['text'][:500]}...\n")
    if tool_results:
        print("=== TOOL RESULTS ===")
        print(json.dumps(tool_results, indent=2))
    return {"answer": final_answer, "retrieved": retrieved, "tools": tool_results, "plan": plan}

# ==== DEMO usage ====
# Replace with your actual PDF file paths
pdf_paths = ["your_doc_1.pdf", "your_doc_2.pdf"]    # <-- change these

# Build index once
idx = rag_over_pdfs(pdf_paths)

# Ask queries
# resp = ask(idx, "Summarize the scooter odometer navigation flow and list calibration steps.")
# (uncomment last line to run a question)
