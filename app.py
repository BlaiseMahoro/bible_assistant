import os
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── Mode config ───────────────────────────────────────────────────────────────
MODES = {
    "⚡ Fast": {
        "k": 4,
        "fetch_k": 12,
        "temperature": 0,
        "description": "Quick answers using the top 4 most relevant verses.",
        "system_prompt": (
            "You are a Bible verse lookup tool. "
            "Answer concisely using ONLY the passages in the CONTEXT below. "
            "Cite every verse as Book Chapter:Verse. "
            "If the answer is not in the CONTEXT, say: 'The retrieved passages do not cover this topic.'\n\n"
            "CONTEXT:\n{context}"
        ),
    },
    "🧠 Deep Thinking": {
        "k": 12,
        "fetch_k": 40,
        "temperature": 0,
        "description": "Thorough answers cross-referencing up to 12 passages for complex questions.",
        "system_prompt": (
            "You are a Bible verse lookup tool. "
            "You have NO knowledge of your own. "
            "The ONLY information you may use is the Bible passages listed in the CONTEXT block below. "
            "Rules you must never break:\n"
            "  - Every claim you make must be supported by a passage in the CONTEXT.\n"
            "  - Always quote the exact verse and cite it as Book Chapter:Verse (e.g. John 3:16).\n"
            "  - Cross-reference multiple passages where relevant to give a thorough answer.\n"
            "  - If the CONTEXT does not contain an answer, reply with: "
            "'The retrieved passages do not cover this topic.' — nothing more.\n"
            "  - Never add explanation, theology, or information from outside the CONTEXT.\n\n"
            "CONTEXT:\n{context}"
        ),
    },
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bible Assistant", page_icon="✝", layout="centered")
st.title("Bible Assistant")
st.caption("Ask any question and get scripture-backed answers.")

# ── Sidebar — mode selector ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Mode",
        options=list(MODES.keys()),
        index=0,
        help="Fast: quick answers. Deep Thinking: thorough cross-referencing.",
    )
    st.caption(MODES[mode]["description"])
    st.divider()
    st.markdown("**Model:** llama3")
    st.markdown("**Bible:** KJV (30,682 verses)")

# ── Vectorstore (cached separately so it survives mode changes) ───────────────
@st.cache_resource(show_spinner="Setting up Bible database — this takes ~2 min on first launch...")
def load_vectorstore():
    DB_PATH = "./bible_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    loader = TextLoader("bible_kjv_clean.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = splitter.split_documents(docs)
    return Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=DB_PATH
    )

# ── Build RAG chain for the selected mode ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_chain(_vectorstore, mode_name):
    cfg = MODES[mode_name]
    llm = ChatOllama(model="llama3", temperature=cfg["temperature"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg["system_prompt"]),
        ("human", "{input}"),
    ])
    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cfg["k"], "fetch_k": cfg["fetch_k"]},
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


vectorstore = load_vectorstore()
rag_chain   = load_rag_chain(vectorstore, mode)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Clear chat button ─────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("Clear chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ── Render history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Source passages"):
                for src in msg["sources"]:
                    st.markdown(f"> {src}")

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the Bible..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    cfg = MODES[mode]

    with st.chat_message("assistant"):
        # Step 1 — retrieval
        with st.status("Searching scripture...", expanded=False) as status:
            docs = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": cfg["k"], "fetch_k": cfg["fetch_k"]},
            ).invoke(question)
            status.update(label=f"Found {len(docs)} relevant passages", state="complete")

        # Step 2 — stream answer
        answer_box = st.empty()
        full_answer = ""
        with st.spinner("Generating answer..."):
            for chunk in rag_chain.stream({"input": question}):
                if "answer" in chunk:
                    full_answer += chunk["answer"]
                    answer_box.markdown(full_answer + "▌")
        answer_box.markdown(full_answer)

        # Step 3 — source passages
        sources = [doc.page_content for doc in docs]
        if sources:
            with st.expander("Source passages"):
                for src in sources:
                    st.markdown(f"> {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
