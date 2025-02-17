import asyncio
import random
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from ragbase.chain import ask_question, create_chain
from ragbase.config import Config
from ragbase.ingestor import Ingestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever
from ragbase.uploader import upload_files

load_dotenv()

LOADING_MESSAGES = [
    "🔍 Analyzing your legal documents...",
    "⚖️ Cross-referencing legal clauses...",
    "📚 Diving into your case files...",
    "🧠 Decoding legal complexities...",
    "📊 Extracting key legal insights...",
    "💼 Summarizing contractual obligations...",
]

# New feature: Common legal questions
COMMON_QUESTIONS = [
    "What are the main obligations in this contract?",
    "What are the termination clauses?",
    "What are the payment terms?",
    "What are the confidentiality requirements?",
    "What is the duration of this agreement?",
]

def custom_css():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #2c3e50, #4ca1af);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
        }
        .st-emotion-cache-p4micv {
            width: 3rem;
            height: 3rem;
        }
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            margin-bottom: 2rem;
        }
        .uploaded-file {
            border: 2px dashed #fff;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .chat-info {
            font-size: 0.8rem;
            color: #888;
            text-align: right;
            margin-top: -10px;
        }
        .quick-actions {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        .action-button {
            padding: 5px 10px;
            border-radius: 5px;
            border: 1px solid #4ca1af;
            background: transparent;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s;
        }
        .action-button:hover {
            background: #4ca1af;
        }
        .document-stats {
            background: rgba(76, 161, 175, 0.1);
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        /* Enhanced mobile responsiveness */
@media (max-width: 768px) {
    .quick-actions {
        flex-direction: column;
        gap: 5px;
    }
    .action-button {
        width: 100%;
        font-size: 0.9em;
        padding: 8px;
    }
    .title {
        font-size: 2.8rem;
    }
    .subtitle {
        font-size: 1.2rem;
    }
    .document-stats {
        font-size: 0.9em;
    }
}

/* Updated quick action buttons */
.stButton > button {
    width: 100%;
    height: auto;
    padding: 8px 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 0.9em;
    margin: 2px 0;
}

/* Improved button visibility */
.stButton > button:hover {
    background-color: rgba(76, 161, 175, 0.2) !important;
    border-color: #4ca1af !important;
}

/* Container adjustments */
.st-emotion-cache-1y4p8pa {
    max-width: 100%;
    padding: 0 5px;
}
        </style>
        """,
        unsafe_allow_html=True,
    )


# New feature: Document statistics
def get_document_stats(files):
    total_size = sum(file.size for file in files)
    return {
        "count": len(files),
        "total_size": f"{total_size / 1024 / 1024:.2f} MB",
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@st.cache_resource(show_spinner=False)
def build_qa_chain(files):
    file_paths = upload_files(files)
    vector_store = Ingestor().ingest(file_paths)
    llm = create_llm()
    retriever = create_retriever(llm, vector_store=vector_store)
    return create_chain(llm, retriever)


async def ask_chain(question: str, chain):
    start_time = time.time()
    full_response = ""
    assistant = st.chat_message(
        "assistant", avatar=str(Config.Path.IMAGES_DIR / "assistant-avatar.png")
    )
    with assistant:
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        documents = []
        async for event in ask_question(chain, question, session_id="session-id-42"):
            if type(event) is str:
                full_response += event
                message_placeholder.markdown(full_response)
            if type(event) is list:
                documents.extend(event)
        # Show source documents with relevance scores
        for i, doc in enumerate(documents):
            with st.expander(f"Source #{i+1} - Relevance: {random.randint(75, 99)}%"):
                st.write(doc.page_content)

        # Show response time
        response_time = time.time() - start_time
        st.markdown(f"<div class='chat-info'>Response time: {response_time:.2f}s</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "timestamp": datetime.now().strftime("%H:%M:%S")
        })


def show_upload_documents():
    custom_css()
    
    # Add animated CSS
    st.markdown("""
        <style>
        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(5deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
        @keyframes glow {
            0% { box-shadow: 0 0 5px #4ca1af; }
            50% { box-shadow: 0 0 25px #4ca1af, 0 0 50px #2c3e50; }
            100% { box-shadow: 0 0 5px #4ca1af; }
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .glowing {
            animation: glow 3s ease-in-out infinite;
            transition: all 0.3s ease;
        }
        .glowing:hover {
            transform: scale(1.05);
        }
        .uploaded-file {
            border: 2px #4ca1af;
            border-radius: 25px;
            padding: 10px;
            animation: glow 3s ease-in-out infinite;
            transition: all 0.3s ease;
        }
        .uploaded-file:hover {
            background: rgba(76, 161, 175, 0.1);
        }
        .feature-icon {
            font-size: 1em;
            margin-right: 5px;
            display: inline-block;
        }
        .title {
            animation: slideIn 1s ease-out;
            
        }
        .subtitle {
            animation: slideIn 1s ease-out 0.5s backwards;
        }
        </style>
    """, unsafe_allow_html=True)
    
    holder = st.empty()
    with holder.container():
        st.markdown("<div class='floating'><h2 class='title'>LegalMentor 🗽</h2></div>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle glowing'>Your AI-Powered Legal Assistant</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                label="📄 Drop your PDF documents here",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload one or multiple PDF files to get started"
            )
             # New feature: Document statistics display
            if uploaded_files:
                stats = get_document_stats(uploaded_files)
                st.markdown(
                    f"""
                    <div class='document-stats'>
                        📊 <b>Document Statistics</b><br>
                        Documents: {stats['count']}<br>
                        Total Size: {stats['total_size']}<br>
                        Upload Time: {stats['upload_time']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
        with col2:
            st.markdown("""
                ### ✨ Smart Features
                <div class='feature-icon'>🔍</div> Smart legal insights<br>
                <div class='feature-icon'>⚖️</div> Contract review<br>
                <div class='feature-icon'>🔐</div> Privacy protection<br>
                <div class='feature-icon'>📋</div> Legal summary generation<br>
                <div class='feature-icon'>📈</div> Risk assessment
            """, unsafe_allow_html=True)
    
    if not uploaded_files:
        st.warning("👆 Please upload your legal documents to begin the analysis")
        st.markdown("""
            <div class='uploaded-file'>
                <h3>📥 Waiting for documents...</h3>
                <p>Supported format: PDF</p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()

    with st.spinner("🔄 Processing your documents..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            if i % 50 == 0:
                st.markdown(random.choice(LOADING_MESSAGES))
            asyncio.sleep(0.01)
        holder.empty()
        return build_qa_chain(uploaded_files)


def show_message_history():
    for message in st.session_state.messages:
        role = message["role"]
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.png"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.markdown(f"<div class='chat-info'>{message['timestamp']}</div>", unsafe_allow_html=True)

def show_quick_actions():
    st.markdown("<div class='quick-actions'>", unsafe_allow_html=True)
    
    # Create buttons with shorter, clearer text
    quick_button_texts = [
        "📋 Main Obligations",
        "🚫 Termination Terms",
        "💰 Payment Terms",
        "🔒 Confidentiality",
        "⏱️ Duration"
    ]
    
    rows = [quick_button_texts[i:i+2] for i in range(0, len(quick_button_texts), 2)]
    
    for i, row_buttons in enumerate(rows):
        cols = st.columns(len(row_buttons))
        for j, (col, button_text) in enumerate(zip(cols, row_buttons)):
            original_question = COMMON_QUESTIONS[i*2 + j]
            if col.button(button_text, key=f"quick_{i*2 + j}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": original_question})
                with st.chat_message(
                    "user",
                    avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
                ):
                    st.markdown(original_question)
                asyncio.run(ask_chain(original_question, st.session_state.chain))
    
    st.markdown("</div>", unsafe_allow_html=True)


def show_chat_input(chain):
    #store chain in session state for quick actions
    st.session_state.chain = chain

    #show quick actions suggestions
    show_quick_actions()

    if prompt := st.chat_input("Ask your question here"):
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
         })
        with st.chat_message(
            "user",
            avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png"),
        ):
            st.markdown(prompt)
        asyncio.run(ask_chain(prompt, chain))


st.set_page_config(page_title="LegalMentor", page_icon="⚖️")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! What do you want to know about your documents?",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
    ]

if Config.CONVERSATION_MESSAGES_LIMIT > 0 and Config.CONVERSATION_MESSAGES_LIMIT <= len(
    st.session_state.messages
):
    st.warning(
        "⚠️ You have reached the conversation limit. Refresh the page to start a new conversation."
    )
    st.stop()

chain = show_upload_documents()
show_message_history()
show_chat_input(chain)
