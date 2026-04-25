"""
DeepSeek-1M · Streamlit Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A beautiful, production-ready web interface for the 1M context demos.

Run with:
    streamlit run demo/app.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from deepseek_1m import DeepSeekClient, Session, load, load_pdf, load_mbox

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepSeek-1M · 1M Context Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0a0a0f; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #1e3a5f;
        text-align: center;
    }
    .main-header h1 { color: #60a5fa; font-size: 2.5rem; margin: 0; }
    .main-header p { color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem; }
    .stat-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value { color: #60a5fa; font-size: 1.8rem; font-weight: bold; }
    .stat-label { color: #6b7280; font-size: 0.85rem; }
    .context-bar {
        background: #1f2937;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .context-fill {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    .chat-user {
        background: #1e3a5f;
        border-radius: 12px 12px 4px 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .chat-assistant {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 4px 12px 12px 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    .thinking-box {
        background: #1a1a0a;
        border: 1px solid #3d3d00;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #a3a300;
        font-size: 0.9rem;
    }
    div[data-testid="stChatInput"] > div { background: #111827; border: 1px solid #374151; }
    .stSelectbox > div > div { background: #111827; color: #e2e8f0; }
    .stTextInput > div > div > input { background: #111827; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "context_loaded": False,
        "context_label": "",
        "context_files": 0,
        "context_tokens": 0,
        "session": None,
        "api_key_set": False,
        "total_cost": 0.0,
        "total_tokens_out": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="sk-...",
        help="Get yours at platform.deepseek.com",
        value=os.environ.get("DEEPSEEK_API_KEY", ""),
    )

    model = st.selectbox(
        "Model",
        ["deepseek-v4-flash", "deepseek-v4-pro"],
        help="Flash: fast & cheap · Pro: most capable",
    )

    thinking_mode = st.toggle(
        "🧠 Thinking Mode",
        value=False,
        help="Enable chain-of-thought reasoning (slower but deeper)",
    )

    st.markdown("---")
    st.markdown("## 📥 Load Context")

    source_type = st.radio(
        "Source Type",
        ["GitHub Repo", "Local Folder", "PDF Document", "Email Archive (mbox)"],
        horizontal=False,
    )

    if source_type == "GitHub Repo":
        repo_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
            value="https://github.com/tiangolo/fastapi",
        )
        github_token = st.text_input(
            "GitHub Token (optional)",
            type="password",
            placeholder="For private repos / higher rate limits",
        )
        load_kwargs = {"token": github_token} if github_token else {}
        source_value = repo_url

    elif source_type == "Local Folder":
        folder_path = st.text_input("Folder Path", placeholder="/path/to/your/project")
        load_kwargs = {}
        source_value = folder_path

    elif source_type == "PDF Document":
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        load_kwargs = {}
        source_value = None

    else:  # mbox
        uploaded_mbox = st.file_uploader("Upload .mbox file", type=["mbox"])
        max_emails = st.slider("Max Emails", 100, 5000, 2000, 100)
        load_kwargs = {"max_emails": max_emails}
        source_value = None

    load_btn = st.button("🚀 Load into 1M Context", type="primary", use_container_width=True)

    if load_btn:
        if not api_key:
            st.error("Please enter your DeepSeek API key.")
        else:
            with st.spinner("Loading context..."):
                try:
                    if source_type == "PDF Document" and uploaded_pdf:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                            f.write(uploaded_pdf.read())
                            tmp_path = f.name
                        ctx = load_pdf(tmp_path)
                        os.unlink(tmp_path)

                    elif source_type == "Email Archive (mbox)" and uploaded_mbox:
                        with tempfile.NamedTemporaryFile(suffix=".mbox", delete=False) as f:
                            f.write(uploaded_mbox.read())
                            tmp_path = f.name
                        ctx = load_mbox(tmp_path, **load_kwargs)
                        os.unlink(tmp_path)

                    elif source_value:
                        ctx = load(source_value, **load_kwargs)
                    else:
                        st.error("Please provide a source to load.")
                        st.stop()

                    client = DeepSeekClient(
                        api_key=api_key,
                        model=model,
                        thinking=thinking_mode,
                    )
                    session = Session(client=client, context=ctx)

                    st.session_state.update({
                        "session": session,
                        "context_loaded": True,
                        "context_label": ctx.source_label,
                        "context_files": ctx.total_files,
                        "context_tokens": ctx.total_tokens_estimate,
                        "messages": [],
                        "api_key_set": True,
                    })
                    st.success(f"✓ Loaded {ctx.total_files} files (~{ctx.total_tokens_estimate:,} tokens)")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error loading context: {e}")

    st.markdown("---")

    if st.session_state.context_loaded:
        st.markdown("## 📊 Context Stats")
        tokens = st.session_state.context_tokens
        pct = min((tokens / 1_000_000) * 100, 100)

        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{tokens:,}</div>
            <div class="stat-label">tokens loaded</div>
        </div>
        <div class="context-bar">
            <div class="context-fill" style="width: {pct:.1f}%"></div>
        </div>
        <div style="color: #6b7280; font-size: 0.8rem; text-align: right;">{pct:.1f}% of 1M window</div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Source:** `{st.session_state.context_label}`")
        st.markdown(f"**Files:** {st.session_state.context_files:,}")
        st.markdown(f"**Total cost:** ${st.session_state.total_cost:.4f}")

        if st.button("🗑 Clear Context", use_container_width=True):
            for k in ["messages", "context_loaded", "context_label",
                      "context_files", "context_tokens", "session"]:
                st.session_state[k] = None if k == "session" else ([] if k == "messages" else False if k == "context_loaded" else "" if isinstance(st.session_state[k], str) else 0)
            st.rerun()


# ── Main content ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚀 DeepSeek-1M</h1>
    <p>Chat with your entire codebase, documents, or email archive — up to 1,000,000 tokens at once</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.context_loaded:
    # Landing state
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">1M</div>
            <div class="stat-label">token context window</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">~750K</div>
            <div class="stat-label">words in one prompt</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">~6</div>
            <div class="stat-label">average novels at once</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">0</div>
            <div class="stat-label">chunking or RAG needed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👈 Load a source from the sidebar to get started")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🗂 Codebase Chat**
        Load any GitHub repo and ask:
        - "Where is the authentication logic?"
        - "Explain the overall architecture"
        - "Find all API endpoints"
        - "What does this function do?"
        """)
        st.markdown("""
        **📚 Document Analysis**
        Load any PDF and ask:
        - "Summarize chapter 3"
        - "Find contradictions"
        - "Extract all key figures"
        - "What are the main arguments?"
        """)
    with col2:
        st.markdown("""
        **📧 Email Archive**
        Load your mbox archive and ask:
        - "Who do I email most?"
        - "Find unanswered threads"
        - "Summarize Q3 discussions"
        - "Any missed follow-ups?"
        """)
        st.markdown("""
        **🔍 How it works**
        1. Set your DeepSeek API key
        2. Choose a source type
        3. Load into 1M context
        4. Chat freely — no limits

        *No embeddings. No chunking. No retrieval.
        Just DeepSeek-V4 holding it all in memory.*
        """)
else:
    # Chat interface
    chat_col, info_col = st.columns([3, 1])

    with info_col:
        st.markdown("### 💡 Suggested prompts")
        if "github" in st.session_state.context_label or st.session_state.context_files > 5:
            prompts = [
                "Explain the overall architecture",
                "How is authentication handled?",
                "Find all API endpoints",
                "What tests exist?",
                "What are the main dependencies?",
            ]
        elif ".pdf" in st.session_state.context_label or "pdf" in st.session_state.context_label:
            prompts = [
                "Give me a high-level summary",
                "What are the key findings?",
                "Find any contradictions",
                "Extract all statistics and numbers",
                "What questions does this leave unanswered?",
            ]
        else:
            prompts = [
                "What patterns do you see?",
                "Summarize everything",
                "Find the most important items",
                "What's missing or concerning?",
            ]

        for p in prompts:
            if st.button(p, use_container_width=True, key=f"prompt_{p}"):
                st.session_state.messages.append({"role": "user", "content": p})
                st.rerun()

    with chat_col:
        # Display message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    if msg.get("thinking"):
                        with st.expander("🧠 Thinking...", expanded=False):
                            st.markdown(msg["thinking"])
                    st.markdown(msg["content"])
                    if msg.get("cost"):
                        st.caption(
                            f"↳ {msg.get('tokens_out', 0):,} tokens · "
                            f"${msg['cost']:.4f} · {msg.get('elapsed', 0):.1f}s"
                        )

        # Check if we need to process a new message (from prompt buttons)
        pending = (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and (
                len(st.session_state.messages) == 1
                or st.session_state.messages[-2]["role"] != "user"
            )
        )

        # Chat input
        if user_input := st.chat_input("Ask anything about your loaded context..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            pending = True
            st.rerun()

        if pending and st.session_state.session:
            last_user = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("DeepSeek-V4 is thinking..."):
                    try:
                        start = time.perf_counter()
                        response = st.session_state.session.ask(last_user, stream=False)
                        elapsed = time.perf_counter() - start

                        # Get last turn stats
                        session = st.session_state.session
                        last_turns = [t for t in session.history if t.role == "assistant"]
                        cost = 0.0
                        tokens_out = 0
                        thinking_text = None

                        if last_turns:
                            lt = last_turns[-1]
                            tokens_out = lt.tokens_out
                            thinking_text = lt.thinking

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "thinking": thinking_text,
                            "tokens_out": tokens_out,
                            "cost": cost,
                            "elapsed": elapsed,
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
