import faiss  # Used for similarity search.
import re
import numpy as np  # Used for numerical operations.
from groq import Groq  # Connects to Groq LLM.
from pypdf import PdfReader  # Reads PDF files.
from sentence_transformers import SentenceTransformer  # Generates text embeddings.
from pathlib import Path


# Files that are not heist case material and would only add noise.
EXCLUDED_PREFIXES = ("GA-2003-72-RES-07",)


def pretty_source(filename):
    # 'Antwerp_diamond_heist (2).pdf' -> 'Antwerp diamond heist'
    name = Path(filename).stem
    name = re.sub(r"\s*\(\d+\)$", "", name)
    return name.replace("_", " ")


# 1. LOAD PDF
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))  # Opens the PDF file.
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""  # Extracts text from each page.
        page_text = page_text.strip()

        if page_text:
            pages.append({
                "text": page_text,
                "page": page_number
            })

    if not pages:
        raise ValueError(f"No readable text found in {pdf_path.name}")

    return pages


# 2. CREATE CHUNKS
def create_chunks(full_text, chunk_size=1000, overlap=200,
                  source="Unknown", page=1):
    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "source": source,
                "page": page
            })

        if end >= len(full_text):
            break

        start = end - overlap

    return chunks


# 3. BUILD VECTORS
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot apply embeddings on empty chunks")

    embed_model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    embed_vectors = embed_model.encode(
        texts,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(embed_vectors, dtype="float32")
    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return embed_model, index


# 4. LOAD BASE
def load_base(pdf_path="datafiles"):
    chunks = []
    pdf_files = [
        p for p in sorted(Path(pdf_path).glob("*.pdf"))
        if not p.name.startswith(EXCLUDED_PREFIXES)
    ]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{pdf_path}'. "
            "Keep all your case PDFs inside the data folder."
        )

    for pdf in pdf_files:
        pages = load_pdf(pdf)

        for page_data in pages:
            page_chunks = create_chunks(
                page_data["text"],
                source=pdf.name,
                page=page_data["page"]
            )
            chunks.extend(page_chunks)

    # Build vectors only once after loading all PDFs.
    embed_model, index = build_vectors(chunks)

    return index, embed_model, chunks


# 5. RETRIEVE CONTENT
# Retrieval is spread across documents: the strongest chunks overall are kept,
# plus the best chunk of every other case file that is reasonably relevant
# (so broad questions cover many cases, specific ones stay focused).
def retrieve_content(question, embed_model, chunks, index,
                     num_chunks=14, core_chunks=6, min_ratio=0.7):
    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_vector = embed_model.encode([question], convert_to_numpy=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    faiss.normalize_L2(query_vector)

    # Score every chunk (the corpus is small), best first.
    sim_scores, vec_indices = index.search(query_vector, len(chunks))
    ranked = [
        (float(sc), int(ix))
        for sc, ix in zip(sim_scores[0], vec_indices[0]) if ix != -1
    ]

    top_score = ranked[0][0]
    selected, chosen = [], set()

    # Pass 1: the strongest chunks overall (deep context for specific questions).
    for score, ix in ranked[:core_chunks]:
        selected.append((score, ix))
        chosen.add(ix)
    covered = {chunks[ix]["source"] for _, ix in selected}

    # Pass 2: best chunk of every other case file that is reasonably relevant
    # (so broad questions are not answered from just one or two cases).
    for score, ix in ranked:
        if len(selected) >= num_chunks:
            break
        src = chunks[ix]["source"]
        if src not in covered and score >= min_ratio * top_score:
            selected.append((score, ix))
            chosen.add(ix)
            covered.add(src)

    selected.sort(key=lambda x: x[0], reverse=True)

    background, sources = "", []
    for score, ix in selected:
        chunk = chunks[ix]
        case = pretty_source(chunk["source"])
        background += (
            f"Case file: {case}, Page: {chunk['page']}\n"
            f"{chunk['text']}\n\n"
        )
        label = f"{case} (page {chunk['page']})"
        if label not in sources:
            sources.append(label)

    return background, sources


# 6. ASK RAG
def ask_rag(question, embed_model, chunks, index, api_key,
             num_chunks=14, model_name="openai/gpt-oss-120b"):

    if not api_key:
        raise ValueError("API Key Missing")

    background, sources = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks
    )

    prompt = f"""You are a research assistant specializing in major robbery cases.

Answer the user's question using ONLY the document background below.
Do not invent, assume, or add facts from outside the documents.
If the information is unavailable, say "Information Missing".
Clearly distinguish confirmed facts from allegations or reported claims.
When possible, identify the relevant robbery case.
If the question is broad or comparative, cover EVERY relevant case found in the
background, not just the first one or two, and name each case explicitly.
Do not create source references that are not present in the background.

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
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    if sources and "information missing" not in answer.lower():
        answer += "\n\nSources:\n" + "\n".join(
            f"- {source}" for source in sources
        )

    return answer
