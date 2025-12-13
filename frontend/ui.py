"""Streamlit frontend for Vietnamese Legal RAG Assistant."""
import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Trợ lý Pháp lý AI", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Be Vietnam Pro', sans-serif; }
    
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); }
    
    .main-title {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    .subtitle {
        color: #64748b;
        text-align: center;
        font-size: 1em;
        margin-bottom: 2em;
    }
    
    .source-item {
        background: #fff;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        transition: all 0.15s ease;
    }
    
    .source-item:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59,130,246,0.1);
    }
    
    .source-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    
    .source-num {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: #fff;
        min-width: 24px;
        height: 24px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75em;
        font-weight: 600;
    }
    
    .source-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.9em;
        flex: 1;
        line-height: 1.3;
    }
    
    .score-pill {
        font-size: 0.7em;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 12px;
        white-space: nowrap;
    }
    
    .score-high { background: #dcfce7; color: #166534; }
    .score-med { background: #dbeafe; color: #1e40af; }
    .score-low { background: #fef3c7; color: #92400e; }
    
    .source-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 8px;
    }
    
    .meta-chip {
        font-size: 0.7em;
        background: #f1f5f9;
        color: #475569;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .source-excerpt {
        font-size: 0.82em;
        color: #475569;
        line-height: 1.6;
        background: #f8fafc;
        padding: 10px 12px;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        max-height: 120px;
        overflow-y: auto;
    }
    
    .stChatMessage { background: transparent; }
    
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid #e5e7eb;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37,99,235,0.3);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    
    .status-on { background: #10b981; box-shadow: 0 0 6px #10b981; }
    .status-off { background: #ef4444; }
    
    details summary { cursor: pointer; }
    
    .sources-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-title">⚖️ Trợ lý Pháp lý AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hybrid Search + ViRanker Reranking • Powered by Gemini</p>', unsafe_allow_html=True)


def check_api_health() -> bool:
    try:
        return httpx.get(f"{API_BASE_URL}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


def send_chat_request(query: str, history: list, temperature: float) -> dict | None:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_BASE_URL}/chat", json={
                "query": query, "history": history, "temperature": temperature
            })
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        st.error("⏱️ Timeout - Vui lòng thử lại")
    except httpx.HTTPStatusError as e:
        st.error(f"❌ Lỗi API: {e.response.status_code}")
    except Exception as e:
        st.error(f"❌ Lỗi kết nối: {e}")
    return None


def render_sources(sources: list) -> None:
    """Render compact source cards."""
    st.markdown('<div class="sources-container">', unsafe_allow_html=True)
    
    for idx, src in enumerate(sources):
        score = src.get("relevance_score", 0.5)
        title = src.get("title", "Văn bản pháp luật")[:80]
        article_ref = src.get("article_ref", "")
        law_id = src.get("law_id", "")
        content = src.get("content", "")
        
        # Score styling
        if score >= 0.7:
            score_class, score_text = "score-high", f"✓ {score*100:.0f}%"
        elif score >= 0.5:
            score_class, score_text = "score-med", f"{score*100:.0f}%"
        else:
            score_class, score_text = "score-low", f"{score*100:.0f}%"
        
        # Build meta chips
        meta_chips = []
        if article_ref:
            meta_chips.append(f'<span class="meta-chip">📌 {article_ref}</span>')
        if law_id:
            meta_chips.append(f'<span class="meta-chip">📜 {law_id}</span>')
        meta_html = "".join(meta_chips)
        
        st.markdown(f'''
        <div class="source-item">
            <div class="source-header">
                <span class="source-num">{idx + 1}</span>
                <span class="source-title">{title}</span>
                <span class="score-pill {score_class}">{score_text}</span>
            </div>
            <div class="source-meta">{meta_html}</div>
            <div class="source-excerpt">{content}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    api_online = check_api_health()
    dot_class = "status-on" if api_online else "status-off"
    status = "Online" if api_online else "Offline"
    st.markdown(f'<p><span class="status-dot {dot_class}"></span>API: <b>{status}</b></p>', unsafe_allow_html=True)
    
    st.divider()
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    
    st.divider()
    st.caption("**Tech Stack**")
    st.markdown("""
    - 🔢 GreenNode Embedding
    - 🎯 ViRanker Reranking  
    - 🤖 Gemini Flash LLM
    - 🔍 Qdrant Hybrid Search
    """)
    
    st.divider()
    if st.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.api_history = []
        st.rerun()

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Nhập câu hỏi ..."):
    if not api_online:
        st.error("❌ API chưa sẵn sàng")
        st.code("uvicorn src.api.server:app --reload", language="bash")
        st.stop()
    
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.api_history.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm kiếm..."):
            response = send_chat_request(query, st.session_state.api_history, temperature)
        
        if response:
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.api_history.append({"role": "assistant", "content": answer})
            
            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)})", expanded=True):
                    render_sources(sources)
