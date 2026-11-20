
import faiss  # Used for similarity search.
import numpy as np  # Used for numerical operations.
from groq import Groq  # Connects to Groq LLM.
from pypdf import PdfReader  # Reads PDF files.
from sentence_transformers import SentenceTransformer  # Generates text embeddings.
from pathlib import Path


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
def create_chunks(full_text, chunk_size=500, overlap=100,
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
    pdf_files = sorted(Path(pdf_path).glob("*.pdf"))

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
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_vector = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_vector = np.asarray(query_vector, dtype="float32")
    faiss.normalize_L2(query_vector)

    sim_scores, vec_indices = index.search(
        query_vector,
        min(num_chunks, len(chunks))
    )

    background = ""
    sources = []

    for score, chunk_index in zip(sim_scores[0], vec_indices[0]):
        if chunk_index != -1:
            chunk = chunks[int(chunk_index)]

            background += (
                f"Source: {chunk['source']}, "
                f"Page: {chunk['page']}\n"
                f"{chunk['text']}\n\n"
            )

            source_name = f"{chunk['source']} (page {chunk['page']})"

            if source_name not in sources:
                sources.append(source_name)

    return background, sources


# 6. ASK RAG
def ask_rag(question, embed_model, chunks, index, api_key,
             num_chunks=3, model_name="openai/gpt-oss-120b"):

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

    if sources:
        answer += "\n\nSources:\n" + "\n".join(
            f"- {source}" for source in sources
        )

    return answer

