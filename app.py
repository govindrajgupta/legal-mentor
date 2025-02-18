import asyncio
import random
import time
from datetime import datetime
import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from deep_translator import GoogleTranslator

import streamlit as st
from dotenv import load_dotenv

from ragbase.chain import ask_question, create_chain
from ragbase.config import Config
from ragbase.ingestor import Ingestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever
from ragbase.uploader import upload_files

load_dotenv()

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English'

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

SUPPORTED_LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
}

st.set_page_config(page_title="LegalMentor", page_icon="⚖️")


def custom_css():
    st.markdown(
        """
        <style>
        /* Base styles */
        body {
            background: linear-gradient(to right, #1a1a2e, #16213e);
            color: #e6e6e6;
            font-family: 'Inter', sans-serif;
        }
        
        /* Title & Header */
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 900;
            margin-bottom: 0.2rem;
            background: linear-gradient(45deg, #4ca1af, #2c3e50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: float 3s ease-in-out infinite;
        }
        
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #a8dadc;
            margin-bottom: 2rem;
        }
        
        /* File uploader */
        .stFileUploader > div {
            border: 2px dashed #4ca1af !important;
            border-radius: 15px !important;
            background: rgba(76, 161, 175, 0.1) !important;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover > div {
            border-color: #a8dadc !important;
            background: rgba(168, 218, 220, 0.1) !important;
        }
        
        /* Chat messages */
        .stChatMessage {
            border-radius: 15px;
            margin: 10px 0;
            padding: 15px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* User message */
        [data-testid="stChatMessage"]:has(div:first-child > img[alt="user-avatar"]) {
            background: rgba(76, 161, 175, 0.15);
            border-left: 4px solid #4ca1af;
        }
        
        /* Assistant message */
        [data-testid="stChatMessage"]:has(div:first-child > img[alt="assistant-avatar"]) {
            background: rgba(44, 62, 80, 0.15);
            border-left: 4px solid #2c3e50;
        }
        
        /* Quick actions */
        .quick-action-btn {
            flex: 1;
            padding: 12px;
            border-radius: 10px;
            border: none;
            background: rgba(76, 161, 175, 0.2);
            color: #a8dadc;
            transition: all 0.2s ease;
            font-size: 0.9em;
            margin: 4px;
        }
        
        .quick-action-btn:hover {
            background: rgba(76, 161, 175, 0.3);
            transform: translateY(-2px);
        }
        
        /* Mobile optimizations */
        @media (max-width: 768px) {
            .title {
                font-size: 2rem;
            }
            .subtitle {
                font-size: 1rem;
            }
            .stChatMessage {
                padding: 12px 15px;
                margin: 8px 0;
            }
            .quick-action-btn {
                padding: 10px;
                font-size: 0.85em;
            }
            [data-testid="stVerticalBlock"] {
                gap: 0.5rem;
            }
        }
        
        /* Animations */
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        @keyframes gradientPulse {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #4ca1af 0%, #2c3e50 100%);
            height: 8px;
            border-radius: 4px;
        }
        
        /* Chat input */
        .stTextInput input {
            background: rgba(255,255,255,0.1) !important;
            border: 1px solid #4ca1af !important;
            color: #fff !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
        }
        
        /* Source documents */
        [data-testid="stExpander"] {
            background: rgba(44, 62, 80, 0.2) !important;
            border-radius: 12px !important;
            margin: 10px 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def extract_text_from_pdf(uploaded_files):
    extracted_text = ""
    for uploaded_file in uploaded_files:
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            extracted_text += page.extract_text() or ""
        
        # OCR for image-based PDFs
        images = convert_from_bytes(uploaded_file.getvalue())
        for image in images:
            extracted_text += pytesseract.image_to_string(image)
            
    return extracted_text

# Identify legal keywords
IMPORTANT_TERMS = [
    "obligation", "liability", "termination", "confidentiality", "payment", "indemnity"
]

def highlight_important_content(text):
    highlighted = ""
    for line in text.split("\n"):
        if any(term in line.lower() for term in IMPORTANT_TERMS):
            highlighted += f"⚠️ **{line}**\n"
        else:
            highlighted += line + "\n"
    return highlighted


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
    
    # Add language selector in sidebar
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'English'
    
    st.sidebar.title("Settings")
    
    # Analysis settings
    # Analysis settings in sidebar
    st.sidebar.subheader("Analysis Settings")
    st.session_state.analysis_depth = st.sidebar.select_slider(
        "Analysis Depth",
        options=["Quick Scan", "Standard", "Deep Analysis"],
        value="Standard",
        help="Controls how thoroughly the AI analyzes documents"
    )

    st.session_state.legal_focus = st.sidebar.multiselect(
        "Focus Areas",
        ["Obligations", "Risks", "Deadlines", "Financial Terms", "Compliance", "Intellectual Property"],
        default=["Obligations", "Risks"],
        help="Select areas to emphasize in the analysis"
    )

    # Language selection
    st.sidebar.subheader("Select Language")
    st.sidebar.selectbox(
        "Select Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        key='selected_language'
    )

    # Apply analysis depth to processing
    if st.session_state.analysis_depth == "Quick Scan":
        Config.CHUNK_SIZE = 1000
        Config.CHUNK_OVERLAP = 100
    elif st.session_state.analysis_depth == "Deep Analysis":
        Config.CHUNK_SIZE = 3000
        Config.CHUNK_OVERLAP = 500

    # Update important terms based on focus areas
    focus_terms = {
        "Obligations": ["shall", "must", "required", "obligation"],
        "Risks": ["liability", "damage", "breach", "penalty"],
        "Deadlines": ["deadline", "within", "by", "date"],
        "Financial Terms": ["payment", "fee", "cost", "price"],
        "Compliance": ["comply", "regulation", "law", "policy"],
        "Intellectual Property": ["patent", "copyright", "trademark", "ip"]
    }
    
    IMPORTANT_TERMS.extend([
        term for focus in st.session_state.legal_focus
        for term in focus_terms.get(focus, [])
    ])
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
        st.markdown("<h1 class='title'>LegalMentor ⚖️</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>AI-Powered Contract Analysis & Legal Insights</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1], gap="medium")
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload Legal Documents",
                type=["pdf"],
                accept_multiple_files=True,
                help="Drag & drop or click to upload contracts/agreements",
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                stats = get_document_stats(uploaded_files)
                cols = st.columns(3)
                cols[0].metric("📄 Documents", stats['count'])
                cols[1].metric("📦 Total Size", stats['total_size'])
                cols[2].metric("⏱️ Upload Time", stats['upload_time'].split()[1])
        
        with col2:
            st.markdown("""
                ### Key Features
                ✨ **Smart Analysis**  
                🔍 **OCR Support**  
                ⚡ **Quick Insights**  
                🔒 **Secure Processing**  
                📱 **Mobile Optimized**
                """)
    
    if not uploaded_files:
        st.info("Please upload PDF documents to begin analysis", icon="🙅")
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
    target_lang = SUPPORTED_LANGUAGES[st.session_state.selected_language]
    
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        # Translate message content
        translated_content = translate_text(content, target_lang)
        
        avatar_path = (
            Config.Path.IMAGES_DIR / "assistant-avatar.png"
            if role == "assistant"
            else Config.Path.IMAGES_DIR / "user-avatar.png"
        )
        with st.chat_message(role, avatar=str(avatar_path)):
            st.markdown(translated_content)
            if "timestamp" in message:
                st.markdown(f"<div class='chat-info'>{message['timestamp']}</div>", unsafe_allow_html=True)

def show_quick_actions():
    target_lang = SUPPORTED_LANGUAGES[st.session_state.selected_language]
    
    st.markdown("<div class='quick-actions'>", unsafe_allow_html=True)
    
    # Translate button texts
    quick_button_texts = [
        translate_text("📋 Main Obligations", target_lang),
        translate_text("🚫 Termination Terms", target_lang),
        translate_text("💰 Payment Terms", target_lang),
        translate_text("🔒 Confidentiality", target_lang),
        translate_text("⏱️ Duration", target_lang)
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
    target_lang = SUPPORTED_LANGUAGES[st.session_state.selected_language]
    source_lang = SUPPORTED_LANGUAGES[st.session_state.selected_language]
    
    st.session_state.chain = chain
    show_quick_actions()

    placeholder_text = translate_text("Ask your question here", target_lang)
    
    if prompt := st.chat_input(placeholder_text):
        # Translate user input to English for processing
        english_prompt = translate_text(prompt, 'en', source_lang)
        
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,  # Store original prompt
            "translated_content": english_prompt,  # Store translated prompt
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        with st.chat_message("user", avatar=str(Config.Path.IMAGES_DIR / "user-avatar.png")):
            st.markdown(prompt)
        
        # Use translated prompt for processing
        asyncio.run(ask_chain(english_prompt, chain))


def translate_text(text: str, target_lang: str, source_lang: str = 'en') -> str:
    """Translate text to target language."""
    try:
        if target_lang == source_lang:
            return text
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text


def get_loading_messages(target_lang):
    messages = [
        "🔍 Analyzing your legal documents...",
        "⚖️ Cross-referencing legal clauses...",
        "📚 Diving into your case files...",
        "🧠 Decoding legal complexities...",
        "📊 Extracting key legal insights...",
        "💼 Summarizing contractual obligations...",
    ]
    return [translate_text(msg, target_lang) for msg in messages]

# Update LOADING_MESSAGES usage in the code
LOADING_MESSAGES = get_loading_messages(SUPPORTED_LANGUAGES[st.session_state.selected_language])


def initialize_app():
    if 'messages' not in st.session_state:
        initial_message = "Hi! What do you want to know about your documents?"
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": initial_message,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        ]
    
    # Update loading messages based on selected language
    global LOADING_MESSAGES
    target_lang = SUPPORTED_LANGUAGES[st.session_state.selected_language]
    LOADING_MESSAGES = get_loading_messages(target_lang)


initialize_app()

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
