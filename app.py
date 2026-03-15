import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bible Assistant", page_icon="✝", layout="centered")
st.title("Bible Assistant")
st.caption("Ask any question and get scripture-backed answers.")

# ── Load / build the RAG chain (cached so it only runs once per session) ─────
@st.cache_resource(show_spinner="Setting up Bible database — this takes ~2 min on first launch...")
def load_rag_chain():
    DB_PATH = "./bible_db"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        loader = TextLoader("bible_kjv_clean.txt")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=DB_PATH
        )

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, api_key=api_key)

    system_prompt = (
        "You are a Bible verse lookup tool. "
        "You have NO knowledge of your own. "
        "The ONLY information you may use is the Bible passages listed in the CONTEXT block below. "
        "Rules you must never break:\n"
        "  - Every claim you make must be supported by a passage in the CONTEXT.\n"
        "  - Always quote the exact verse and cite it as Book Chapter:Verse (e.g. John 3:16).\n"
        "  - If the CONTEXT does not contain an answer, reply with: "
        "'The retrieved passages do not cover this topic.' — nothing more.\n"
        "  - Never add explanation, theology, or information from outside the CONTEXT.\n\n"
        "CONTEXT:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # MMR (Maximum Marginal Relevance) returns diverse relevant chunks
    # instead of the top-k most similar, reducing redundancy and improving coverage.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 40},
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


rag_chain = load_rag_chain()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the Bible..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching scripture..."):
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]

        st.markdown(answer)

        # Show source passages in an expander
        sources = response.get("context", [])
        if sources:
            with st.expander("Source passages"):
                for doc in sources:
                    st.markdown(f"> {doc.page_content}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
