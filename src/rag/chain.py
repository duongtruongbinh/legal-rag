"""LLM Generation Chain with conversational context."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from src.core.config import settings
from src.rag.retriever import get_hybrid_retriever


CONTEXTUALIZE_PROMPT = """
Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, hãy tóm tắt lại thành một câu hỏi pháp lý hoàn chỉnh.
Mục tiêu: Để hệ thống tìm kiếm văn bản luật có thể hiểu được ngữ cảnh.
Lưu ý:
- Giữ nguyên các từ khóa quan trọng (tên luật, hành vi, mức phạt...).
- KHÔNG trả lời câu hỏi.
"""

QA_SYSTEM_PROMPT = """
Bạn là "Trợ lý Pháp lý AI" thân thiện và am hiểu pháp luật Việt Nam.
Nhiệm vụ của bạn là giải đáp thắc mắc pháp lý cho người dùng phổ thông dựa trên thông tin được cung cấp (Context).

HƯỚNG DẪN TRẢ LỜI:
1.  **Phong cách:** Dùng ngôn ngữ đời thường, dễ hiểu, tránh lạm dụng từ ngữ chuyên môn khô khan. Giọng văn nhẹ nhàng, khách quan nhưng có sự thấu hiểu.
2.  **Cấu trúc câu trả lời:**
    * **Kết luận trước:** Trả lời trực tiếp vào câu hỏi (Được/Không, Có/Không, Mức phạt là bao nhiêu...).
    * **Giải thích:** Diễn giải nội dung quy định một cách mạch lạc.
    * **Cơ sở pháp lý:** Luôn trích dẫn nguồn để người dùng tin tưởng (Ví dụ: "Chi tiết tại Khoản 1, Điều 5...").
3.  **Trình bày:** Sử dụng danh sách gạch đầu dòng (bullet points) và **in đậm** các thông tin quan trọng (như số tiền phạt, số năm tù, điều kiện...) để người đọc dễ nắm bắt.
4.  **Trung thực:** Nếu ngữ cảnh (Context) không có thông tin, hãy nói: "Xin lỗi, hiện tại tôi chưa tìm thấy văn bản quy định cụ thể về vấn đề này trong cơ sở dữ liệu." Đừng cố gắng bịa ra luật.

---
Dưới đây là các văn bản pháp luật liên quan (Context):
{context}
"""


def get_llm(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    """Get configured LLM instance."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature or settings.llm_temperature,
        google_api_key=settings.google_api_key,
        convert_system_message_to_human=True,
    )


@lru_cache
def get_rag_chain(temperature: float | None = None) -> Runnable:
    """
    Build complete RAG chain with history-aware retrieval.
    
    Args:
        temperature: Optional LLM temperature override.
    
    Returns:
        Configured RAG chain that accepts input and chat_history.
    """
    llm = get_llm(temperature)
    retriever = get_hybrid_retriever()
    
    # Contextualize prompt for history-aware retrieval
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    
    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

