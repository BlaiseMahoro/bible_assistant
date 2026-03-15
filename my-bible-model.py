import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DB_PATH = "./bible_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 1. EMBED & STORE — only rebuild the database if it doesn't exist yet.
#    Delete the bible_db folder manually if you want to re-index.
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print("Loading existing vector database...")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    print("Building vector database from bible_kjv_clean.txt ...")
    # Load the cleaned file (one verse per line, format: "Book Ch:V text")
    loader = TextLoader("bible_kjv_clean.txt")
    docs = loader.load()

    # Each chunk covers ~5 verses; overlap keeps cross-verse context intact.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )
    print(f"Indexed {len(splits)} chunks.")

# 2. LLM SETUP
llm = ChatOllama(model="llama3", temperature=0)

# 3. PROMPT
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

# 4. CHAIN — MMR retrieval for diverse, non-redundant coverage
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 12, "fetch_k": 40},
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. CHAT — change this query to whatever you want to ask
query = "What does the Bible say about Jerusalem?"
response = rag_chain.invoke({"input": query})

print(f"\nAnswer:\n{response['answer']}")
