"""Streamlit Web Application for Japanese RAG Chatbot.

Deployable to Streamlit Community Cloud (https://share.streamlit.io).
Features:
- Real-time token streaming response
- Full Japanese & English bilingual UI
- Multi-turn conversation memory with history-aware citations
- Expandable document sources with exact page references
- Caching of heavy embeddings & BM25 indices
"""

import html
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Community Cloud secret resolution (fallback gracefully if no secrets.toml)
try:
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
except Exception:
    pass

from app.documents import load_or_build_app_index
from app.generator import MODEL, SYSTEM_PROMPT, build_prompt, get_client, topics_rule
from app.retriever import Retriever
from google.genai import errors, types

# Page Configuration
st.set_page_config(
    page_title="日本語RAGチャットボット | Japanese RAG Assistant",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Japanese-first Typography & Styling
st.markdown(
    """
<style>
    /* Clean CJK typography */
    html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput {
        font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans JP", Meiryo, sans-serif !important;
    }
    .main-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a202c;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .badge-tag {
        background: #e53e3e;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .sub-text {
        color: #718096;
        font-size: 0.92rem;
        margin-top: 0.25rem;
    }
    .source-card {
        background: #f7fafc;
        border-left: 3px solid #3182ce;
        padding: 0.5rem 0.75rem;
        margin-top: 0.4rem;
        border-radius: 4px;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .example-btn {
        margin-bottom: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="📚 ドキュメントと検索インデックスを準備中... (Loading knowledge base...)")
def load_rag_engine():
    """Builds or loads the hybrid FAISS + BM25 retriever once across sessions."""
    index = load_or_build_app_index()
    retriever = Retriever(index)
    topics = sorted({c.source.split("/")[0].replace("_", " ") for c in index.chunks})
    return retriever, topics


# Initialize RAG retriever
try:
    retriever, topics = load_rag_engine()
except Exception as e:
    st.error(f"インデックスの初期化に失敗しました / Index failed to load: {e}")
    st.stop()

# A public demo spends the owner's Gemini credit, so cap usage: per browser session per
# minute, and in total per day across all visitors (st.cache_resource is shared by all sessions).
MAX_PER_MINUTE = int(os.getenv("MAX_QUESTIONS_PER_MINUTE", "5"))
MAX_PER_DAY = int(os.getenv("MAX_QUESTIONS_PER_DAY", "300"))


@st.cache_resource
def daily_usage():
    return {"day": "", "count": 0, "lock": threading.Lock()}


def rate_limit_message() -> str | None:
    """None if this question is allowed, otherwise a message explaining the limit."""
    now = time.time()
    recent = [t for t in st.session_state.get("ask_times", []) if now - t < 60]
    if len(recent) >= MAX_PER_MINUTE:
        return "質問が多すぎます。1分ほど待ってください。/ Too many questions. Please wait a minute."
    usage = daily_usage()
    with usage["lock"]:  # sessions run in parallel threads, so update the counter one at a time
        today = time.strftime("%Y-%m-%d")
        if usage["day"] != today:
            usage.update(day=today, count=0)
        if usage["count"] >= MAX_PER_DAY:
            return "本日の質問数の上限に達しました。明日もう一度お試しください。/ The demo's daily limit is reached. Please try again tomorrow."
        usage["count"] += 1
    st.session_state["ask_times"] = recent + [now]
    return None


# Header
st.markdown(
    """
<div class="main-header">
    <div class="main-title">
        <span>🗾 日本語RAGチャットボット</span>
        <span class="badge-tag">RAG with Sources</span>
    </div>
    <div class="sub-text">
        出入国在留管理庁「外国人生活・就労ガイドブック」および日本語Wikipediaに基づいて正確に回答します（日本語・English対応）。
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.subheader("📖 収録ドキュメント / Knowledge Base")
    st.info(
        """
        - **出入国在留管理庁**
          『外国人生活・就労ガイドブック』(全157頁)
          *在留手続、住居、労働、年金・保険、防災、教育等*
        - **JSQuAD 日本語Wikipedia**
          *日本の地理、気象(梅雨等)、文化、歴史 59記事*
        """
    )

    st.markdown("---")
    st.subheader("💡 質問の例 / Example Questions")

    examples = [
        "在留カードをなくしたらどうすればいいですか？",
        "What number do I call for an ambulance?",
        "転職しました。入管に届出は必要ですか？",
        "国民年金の脱退一時金は誰が受け取れますか？",
        "子どもの小学校入学の手続きについて教えて",
        "梅雨の時期の特徴は何ですか？",
    ]

    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["user_input_override"] = ex

    st.markdown("---")
    st.markdown(f"**🤖 モデル**: `{MODEL}`")
    st.markdown(f"**📊 チャンク総数**: `{len(retriever.chunks):,} chunks`")

    if st.button("🗑️ 会話をリセット (Clear Chat)", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "こんにちは！日本の生活・就労ガイドブックや公式情報について何でも質問してください。\n(Ask me anything in Japanese or English; answers are grounded with official document citations.)",
            "sources": [],
        }
    ]

# Display Existing Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 引用元・根拠ドキュメント ({len(msg['sources'])}件)"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""<div class="source-card">
                        <b>[{s['number']}] {html.escape(s['source'])}</b><br>
                        {html.escape(s['text'])}
                        </div>""",
                        unsafe_allow_html=True,
                    )


# Function to Stream Generation from Gemini
def generate_streaming(question: str, top_chunks, history: list[str]):
    system = SYSTEM_PROMPT + topics_rule(topics)
    prompt = build_prompt(question, top_chunks, history)
    client = get_client()

    for attempt in range(3):
        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except errors.ClientError as ce:
            if getattr(ce, "code", None) == 429 or "429" in str(ce):
                if attempt < 2:
                    import time

                    time.sleep(3 * (attempt + 1))
                    continue
            raise ce
        except errors.ServerError as se:
            if attempt < 2:
                import time

                time.sleep(2 * (attempt + 1))
                continue
            raise se


# Input handling (from user type or example button click)
prompt_input = st.chat_input("質問を入力してください / Ask a question in Japanese or English...")
override = st.session_state.pop("user_input_override", None)
user_prompt = override or prompt_input

limit_message = rate_limit_message() if user_prompt else None
if limit_message:
    st.warning(limit_message)
elif user_prompt:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare Assistant Response
    with st.chat_message("assistant"):
        # 1. Retrieve relevant chunks
        with st.spinner("ドキュメントを検索中... / Searching documents..."):
            top = retriever.search(user_prompt, k=5)
            sources = [{"number": i, "source": c.source, "text": c.text} for i, c in enumerate(top, start=1)]

        # 2. Gather conversation history for context
        prior_history = [
            m["content"]
            for m in st.session_state.messages
            if m["role"] == "user" and m["content"] != user_prompt
        ]

        # 3. Stream Response with Fallback
        try:
            full_response = st.write_stream(generate_streaming(user_prompt, top, prior_history))
        except Exception as e:
            full_response = f"⚠️ 回答の生成中にエラーが発生しました: {e}"
            st.error(full_response)

        # 4. Show Document Sources
        if sources:
            with st.expander(f"📚 引用元・根拠ドキュメント ({len(sources)}件)"):
                for s in sources:
                    st.markdown(
                        f"""<div class="source-card">
                        <b>[{s['number']}] {html.escape(s['source'])}</b><br>
                        {html.escape(s['text'])}
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        }
    )
