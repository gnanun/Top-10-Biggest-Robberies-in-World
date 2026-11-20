
import streamlit as st
from rag_engine import load_base, ask_rag


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Top 10 Robbery Cases",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# 2. CUSTOM CSS
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }

    .sub-title {
        text-align: center;
        color: gray;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)


# 3. SIDEBAR
with st.sidebar:

    st.title("Project Information")

    info, pipeline = st.tabs(["Info", "Pipeline"])

    with info:

        st.subheader("Technology Used")

        st.write("""
        - Python
        - Streamlit
        - PyPDF
        - Sentence Transformers
        - FAISS
        - Groq LLM
        """)

        st.divider()

        st.subheader("Description")

        st.write("""
        This application uses Retrieval-Augmented Generation
        to answer questions based on ten major robbery cases.

        Answers are generated using information retrieved
        from the uploaded PDF documents.
        """)

        st.divider()

        st.subheader("Team Details")

        st.write("Developed by: Gnanesh N")

        # Add other team members here if required.

    with pipeline:

        st.subheader("RAG Pipeline")

        st.write("""
        1. Load PDF Documents
        2. Extract Text
        3. Create Text Chunks
        4. Generate Embeddings
        5. Store Vectors in FAISS
        6. Retrieve Relevant Content
        7. Generate Answers using Groq
        """)


# 4. MAIN TITLE
st.markdown(
    '<div class="main-title">Top 10 Robbery Cases</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">RAG-Based Research Assistant</div>',
    unsafe_allow_html=True
)

st.write("")


# 5. GROQ API KEY
try:
    mykey = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    mykey = None


# 6. INITIALIZE RAG ENGINE
@st.cache_resource
def initiate_rag():
    return load_base("datafiles")


# 7. LOAD INDEX, EMBEDDING MODEL AND CHUNKS
try:
    index, embed_model, chunks = initiate_rag()

except Exception as error:
    st.error(f"Unable to load RAG engine: {error}")
    st.stop()


# 8. INITIALIZE CHAT HISTORY
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []


# 9. DISPLAY PREVIOUS CHAT MESSAGES
for qa in st.session_state.chat_hist:

    with st.chat_message(qa["role"]):
        st.write(qa["content"])


# 10. ACCEPT USER QUESTION
question = st.chat_input(
    "Ask anything about the top 10 robbery cases..."
)


# 11. GENERATE ANSWER
if question:

    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching robbery case documents..."):

            try:
                if not mykey:
                    raise ValueError(
                        "Groq API key is missing. Configure GROQ_API_KEY in Streamlit Secrets."
                    )

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=mykey
                )

                st.write(answer)

            except Exception as error:
                answer = f"Error: {error}"
                st.error(answer)

            st.session_state.chat_hist.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


# 12. CLEAR CHAT HISTORY
btn = st.button("Clear Chat")

if btn:

    st.session_state.chat_hist = []

    st.success("Chat History Cleared Successfully!")

    st.rerun()

