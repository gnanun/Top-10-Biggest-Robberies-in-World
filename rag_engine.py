import re
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
# PDFs whose filenames contain any of these substrings are skipped.
EXCLUDE_PATTERNS = (
    "interpol",
    "ecb",
    "co-operation agreement",
    "cooperation agreement",
)

DEFAULT_CHUNK_SIZE = 350
DEFAULT_OVERLAP = 80
DEFAULT_NUM_CHUNKS = 10


# ---------------------------------------------------------------------------
# 1. CLEAN TABLE MARKUP
# ---------------------------------------------------------------------------
def _clean_table_markup(text: str) -> str:
    """Turn <table><tr><td>...</td></tr></table> noise into readable text."""
    text = re.sub(r"</t[dh]>", " | ", text)
    text = re.sub(r"</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. LOAD PDF
# ---------------------------------------------------------------------------
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = _clean_table_markup(page_text)

        if page_text:
            pages.append({
                "text": page_text,
                "page": page_number,
            })

    if not pages:
        raise ValueError(f"No readable text found in {pdf_path.name}")

    return pages


# ---------------------------------------------------------------------------
# 3. CREATE CHUNKS
# ---------------------------------------------------------------------------
def create_chunks(full_text, chunk_size=DEFAULT_CHUNK_SIZE,
                  overlap=DEFAULT_OVERLAP, source="Unknown", page=1):
    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "source": source,
                "page": page,
            })

        if end >= len(full_text):
            break

        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# 4. BUILD VECTORS
# ---------------------------------------------------------------------------
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot apply embeddings on empty chunks")

    embed_model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]

    embed_vectors = embed_model.encode(texts, convert_to_numpy=True)
    embed_vectors = np.asarray(embed_vectors, dtype="float32")
    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return embed_model, index


# ---------------------------------------------------------------------------
# 5. LOAD BASE
# ---------------------------------------------------------------------------
def load_base(pdf_path="datafiles"):
    chunks = []
    pdf_files = sorted(Path(pdf_path).glob("*.pdf"))

    # Filter out off-topic documents (ECB/Interpol resolution, etc.)
    pdf_files = [
        p for p in pdf_files
        if not any(pat in p.name.lower() for pat in EXCLUDE_PATTERNS)
    ]

    if not pdf_files:
        raise FileNotFoundError(
            f"No relevant PDF files found in '{pdf_path}'. "
            "Check EXCLUDE_PATTERNS or add your case PDFs."
        )

    for pdf in pdf_files:
        pages = load_pdf(pdf)
        for page_data in pages:
            page_chunks = create_chunks(
                page_data["text"],
                source=pdf.name,
                page=page_data["page"],
            )
            chunks.extend(page_chunks)

    embed_model, index = build_vectors(chunks)
    return index, embed_model, chunks


# ---------------------------------------------------------------------------
# 6. RETRIEVE CONTENT (with dedupe)
# ---------------------------------------------------------------------------
def retrieve_content(question, embed_model, chunks, index,
                     num_chunks=DEFAULT_NUM_CHUNKS):
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_vector = embed_model.encode([question], convert_to_numpy=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    faiss.normalize_L2(query_vector)

    # Over-fetch so we can dedupe down to num_chunks unique results.
    fetch_k = min(num_chunks * 2, len(chunks))
    sim_scores, vec_indices = index.search(query_vector, fetch_k)

    background = ""
    sources = []
    seen = set()

    for score, chunk_index in zip(sim_scores[0], vec_indices[0]):
        if chunk_index == -1:
            continue

        chunk = chunks[int(chunk_index)]
        key = (chunk["source"], chunk["page"], chunk["text"][:80])

        if key in seen:
            continue
        seen.add(key)

        background += (
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"{chunk['text']}\n\n"
        )

        source_name = f"{chunk['source']} (page {chunk['page']})"
        if source_name not in sources:
            sources.append(source_name)

        if len(seen) >= num_chunks:
            break

    return background, sources


# ---------------------------------------------------------------------------
# 7. ASK RAG
# ---------------------------------------------------------------------------
def ask_rag(question, embed_model, chunks, index, api_key,
            num_chunks=DEFAULT_NUM_CHUNKS,
            model_name="openai/gpt-oss-120b"):

    if not api_key:
        raise ValueError("API Key Missing")

    background, sources = retrieve_content(
        question, embed_model, chunks, index, num_chunks
    )

    prompt = f"""You are a research assistant specializing in major robbery cases.

Answer the user's question using ONLY the document background below.
Do not invent, assume, or add facts from outside the documents.
If the information is unavailable, say "Information Missing".

Rules for conflicting or uncertain data:
- If sources disagree on a figure (value, date, count), state the range
  and note that estimates vary between sources. Do not silently pick one.
- Clearly distinguish confirmed facts from allegations or reported claims.
- For recent or ongoing cases (e.g. the 2025 Louvre theft), state the
  status as of the source document and note that it may be out of date.
- When possible, identify the relevant robbery case by name.
- Do not create source references that are not present in the background.

Document Background:
{background}

User Question:
{question}

Answer:
"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document-grounded research assistant. "
                    "Use only the supplied context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content.strip()

    if sources:
        answer += "\n\nSources:\n" + "\n".join(
            f"- {source}" for source in sources
        )

    return answer
