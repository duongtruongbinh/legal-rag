"""Streamlit frontend - Thin client consuming FastAPI backend."""
import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Legal Assistant Pro", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    .stApp {
        background-color: #f8f9fa;
        color: #1f2937;
    }
    
    .main-title {
        font-family: 'Crimson Pro', serif;
        font-size: 3em;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0.2em;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748b;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 2.5em;
        font-weight: 400;
    }
    
    .source-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    
    .score-badge {
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .meta-tag {
        font-size: 0.8em;
        color: #475569;
        background: #f1f5f9;
        padding: 4px 8px;
        border-radius: 4px;
        margin-right: 8px;
        font-family: 'Inter', sans-serif;
        border: 1px solid #e2e8f0;
        font-weight: 500;
    }
    
    .source-title {
        font-family: 'Crimson Pro', serif;
        font-weight: 700;
        font-size: 1.15em;
        color: #0f172a;
        letter-spacing: -0.01em;
    }
    
    .source-content {
        font-size: 0.95em;
        color: #334155;
        margin-top: 12px;
        line-height: 1.6;
        background: #f8fafc;
        padding: 14px;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        border-left: 3px solid #94a3b8;
    }
    
    .stChatMessage {
        font-family: 'Inter', sans-serif;
        background-color: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #374151;
    }
    
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: #10b981;
        box-shadow: 0 0 4px #10b981;
    }
    
    .status-offline {
        background: #ef4444;
        box-shadow: 0 0 4px #ef4444;
    }
    
    hr {
        margin: 1.5em 0;
        border-color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-title">⚖️ Trợ lý Pháp lý Thông minh</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hệ thống RAG Hybrid Search • Powered by Gemini & Qdrant</p>', unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if backend API is available."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


def send_chat_request(query: str, history: list, temperature: float) -> dict | None:
    """Send chat request to backend API."""
    payload = {
        "query": query,
        "history": history,
        "temperature": temperature,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{API_BASE_URL}/chat", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"❌ API Error: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")
        return None


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []

with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    api_online = check_api_health()
    status_class = "status-online" if api_online else "status-offline"
    status_text = "Online" if api_online else "Offline"
    st.markdown(
        f'<div style="color: #374151; margin-bottom: 10px;"><span class="status-indicator {status_class}"></span>API Status: <b>{status_text}</b></div>',
        unsafe_allow_html=True,
    )
    
    st.divider()
    temperature = st.slider("Độ sáng tạo (Temperature)", 0.0, 1.0, 0.1, 0.05)
    
    st.divider()
    st.markdown("""
    **🔧 Tech Stack:**
    - Dense: GreenNode-Large-VN
    - Sparse: BM25 (FastEmbed)
    - LLM: Gemini Flash
    - Search: Hybrid (Dense + Sparse)
    """)
    
    st.divider()
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.api_history = []
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Nhập câu hỏi pháp lý... (Ví dụ: Quy định về nồng độ cồn)"):
    if not api_online:
        st.error("❌ Backend API is not available. Please start the server first.")
        st.code("uv run uvicorn src.api.server:app --reload", language="bash")
        st.stop()
    
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.api_history.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang phân tích văn bản luật..."):
            response = send_chat_request(
                query=query,
                history=st.session_state.api_history,
                temperature=temperature,
            )
        
        if response:
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.api_history.append({"role": "assistant", "content": answer})
            
            if sources:
                with st.expander("📚 Nguồn tham khảo & Minh chứng", expanded=False):
                    for idx, src in enumerate(sources):
                        score = src.get("relevance_score") or 0.0
                        
                        if score > 0.7:
                            color = "#059669"
                            bg_badge = "#059669"
                        elif score > 0.4:
                            color = "#d97706"
                            bg_badge = "#d97706"
                        else:
                            color = "#dc2626"
                            bg_badge = "#dc2626"
                        
                        title = src.get("title", "N/A")
                        doc_id = src.get("doc_id", "N/A")
                        law_id = src.get("law_id", "N/A")
                        content = src.get("content", "")
                        
                        score_display = f"{score * 100:.1f}%" if score else "N/A"
                        
                        st.markdown(f"""
                        <div class="source-card" style="border-left: 5px solid {color};">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <span class="source-title">{idx + 1}. {title}</span>
                                <span class="score-badge" style="background: {bg_badge};">{score_display}</span>
                            </div>
                            <div style="margin: 10px 0;">
                                <span class="meta-tag">🆔 {doc_id}</span>
                                <span class="meta-tag">📜 {law_id}</span>
                            </div>
                            <div class="source-content">{content}</div>
                        </div>
                        """, unsafe_allow_html=True)