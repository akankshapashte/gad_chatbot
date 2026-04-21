import os
import streamlit as st
import nest_asyncio

from langdetect import detect
from deep_translator import GoogleTranslator

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

nest_asyncio.apply()

st.set_page_config(page_title="GAD Chatbot")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Noto Sans Devanagari", "Mangal", sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("General Administration Department Chatbot")

PERSIST_DIR = "./storage"

@st.cache_resource
def initialize_engine():

    if not os.path.exists(PERSIST_DIR):
        st.error("Run create_index.py first")
        st.stop()

    llm = Ollama(
        model="mistral",
        temperature=0.1,
        request_timeout=500
    )
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-small"
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=PERSIST_DIR
    )

    index = load_index_from_storage(storage_context)

    vector_retriever = index.as_retriever(similarity_top_k=5)

    nodes = list(index.docstore.docs.values())

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3
    )

    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=3,
        num_queries=3,
        mode="reciprocal_rerank",
        use_async=False 
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_mode="tree_summarize"
    )

    return query_engine


query_engine = initialize_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_query := st.chat_input("Ask your question."):

    # Store user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    try:
        lang = detect(user_query)
    except:
        lang = "en"

    if lang == "mr":
        try:
            query_for_rag = GoogleTranslator(source='auto', target='en').translate(user_query)
        except:
            query_for_rag = user_query
    else:
        query_for_rag = user_query

    formatted_query = "query: " + query_for_rag.strip()

    # ---------------- RESPONSE ----------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = query_engine.query(formatted_query)
                answer_en = response.response.strip()

                if not answer_en:
                    answer_en = "No relevant answer found in documents."

            except Exception as e:
                answer_en = f"Error: {str(e)}"

            if lang == "mr":
                try:
                    answer = GoogleTranslator(source='en', target='mr').translate(answer_en)
                except:
                    answer = answer_en
            else:
                answer = answer_en

            st.markdown(answer)

    
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })