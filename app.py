import streamlit as st
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import tempfile
import os
from groq import Groq

# ── New imports for web support ──
from langchain_community.document_loaders import WebBaseLoader

# Page setup
st.set_page_config(page_title="RAG PDF & Web Voice Assistant", layout="wide")
st.title("📄🌐 RAG Voice Assistant – PDF or Website (Groq)")
st.info("Upload PDF **or** enter Website URL → Ask via text or voice → Get text + spoken answer")

# Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_source" not in st.session_state:
    st.session_state.current_source = None

# ────────────────────────────────────────────────
# INPUT SECTION – PDF or URL
# ────────────────────────────────────────────────
col1, col2 = st.columns([3, 4])

with col1:
    st.subheader("Option 1 – PDF")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], key="pdf_uploader")

with col2:
    st.subheader("Option 2 – Website")
    url_input = st.text_input("Enter website URL[](https://...)", 
                              placeholder="https://example.com/article",
                              key="url_input")
    btn_web = st.button("Load Website")

# Status area
source_status = st.empty()

# ── Process PDF ──
if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    st.write(f"PDF uploaded - size: {len(pdf_bytes):,} bytes")
    
    if len(pdf_bytes) < 200:
        st.error("File too small or empty.")
    else:
        try:
            with st.spinner("Processing PDF..."):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                docs_list = []
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text("text")
                    docs_list.append(
                        Document(
                            page_content=text,
                            metadata={"source": uploaded_file.name, "page": page_num + 1}
                        )
                    )
                doc.close()

                if not docs_list or all(len(d.page_content.strip()) == 0 for d in docs_list):
                    st.warning("No readable text extracted (scanned PDF?).")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                    splits = text_splitter.split_documents(docs_list)
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_source = f"PDF: {uploaded_file.name}"
                    source_status.success(f"✅ Ready! Source: {st.session_state.current_source} ({len(docs_list)} pages)")
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")

# ── Process Website ──
if btn_web and url_input.strip():
    url = url_input.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        try:
            with st.spinner(f"Loading & processing {url} ..."):
                loader = WebBaseLoader(url)
                web_docs = loader.load()  # returns list[Document]
                
                if not web_docs or not web_docs[0].page_content.strip():
                    st.warning("No readable content found on this page.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                    splits = text_splitter.split_documents(web_docs)
                    
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_source = f"Website: {url}"
                    source_status.success(f"✅ Ready! Source: {st.session_state.current_source} ({len(web_docs)} page(s) loaded)")
        except Exception as e:
            st.error(f"Failed to load/process website: {str(e)}\n\nTip: Some sites block scraping or require JavaScript.")

# ────────────────────────────────────────────────
# Chat & Voice section
# ────────────────────────────────────────────────
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Answer based only on the provided context. Be concise, accurate and helpful.
Context:
{context}

Question: {question}

Answer:"""
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    st.markdown("---")
    st.subheader(f"Ask about: {st.session_state.current_source}")
    
    # Input (text + voice)
    user_input = st.chat_input(
        placeholder="Ask question (type or speak)...",
        accept_audio=True
    )
    
    if user_input is not None:
        st.write("Input received!")
        question = ""
        
        if hasattr(user_input, "text") and user_input.text:
            question = user_input.text.strip()
            st.write(f"**Typed:** {question}")
        
        if hasattr(user_input, "audio") and user_input.audio:
            audio_upload = user_input.audio
            audio_bytes = audio_upload.getvalue()
            st.write(f"Audio captured - size: {len(audio_bytes):,} bytes")
            
            with st.spinner("Transcribing voice..."):
                try:
                    transcription = groq_client.audio.transcriptions.create(
                        file=("audio.wav", audio_bytes, "audio/wav"),
                        model="whisper-large-v3-turbo",
                        response_format="text",
                        language="en"
                    )
                    question = (transcription or "").strip()
                    if question:
                        st.caption(f"**You said:** {question}")
                    else:
                        st.warning("Transcription returned empty result.")
                except Exception as e:
                    st.error(f"Transcription failed: {str(e)}")
        
        if not question:
            st.info("No question detected. Try typing or speaking more clearly (3–8 seconds).")
        
        if question:
            with st.spinner("Generating answer..."):
                try:
                    response = rag_chain.invoke(question)
                    answer_text = response.content.strip()
                    
                    st.subheader("Answer")
                    st.markdown(answer_text)
                    
                    with st.spinner("Creating voice..."):
                        tts = gTTS(text=answer_text, lang='en')
                        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(tmp_audio.name)
                        
                        st.audio(tmp_audio.name, format="audio/mp3")
                        
                        with open(tmp_audio.name, "rb") as f:
                            st.download_button(
                                label="Download MP3",
                                data=f,
                                file_name="answer.mp3",
                                mime="audio/mp3"
                            )
                    
                    # Cleanup
                    if 'tmp_audio' in locals():
                        try:
                            os.unlink(tmp_audio.name)
                        except:
                            pass
                
                except Exception as e:
                    st.error(f"Answer generation failed: {str(e)}")

else:
    st.info("Please upload a PDF or load a website first.")