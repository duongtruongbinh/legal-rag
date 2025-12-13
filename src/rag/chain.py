"""LLM Generation Chain with template-based prompts."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from src.core.config import settings
from src.rag.retriever import get_hybrid_retriever


def _load_template(name: str) -> str:
    """Load prompt template from file."""
    path = settings.templates_dir / name
    return path.read_text(encoding="utf-8")


@lru_cache
def _get_contextualize_prompt() -> ChatPromptTemplate:
    """Build contextualize prompt for history-aware retrieval."""
    system = _load_template("contextualize.jinja")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


@lru_cache
def _get_qa_prompt() -> ChatPromptTemplate:
    """Build QA prompt with context variable."""
    system = _load_template("qa_system.jinja")
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def _create_llm(temperature: float | None = None, streaming: bool = False) -> ChatGoogleGenerativeAI:
    """Create LLM instance."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature or settings.llm_temperature,
        google_api_key=settings.google_api_key,
        convert_system_message_to_human=True,
        streaming=streaming,
    )


def _build_chain(llm: ChatGoogleGenerativeAI) -> Runnable:
    """Build RAG chain with given LLM."""
    retriever = get_hybrid_retriever()
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, _get_contextualize_prompt()
    )
    
    qa_chain = create_stuff_documents_chain(llm, _get_qa_prompt())
    
    return create_retrieval_chain(history_aware_retriever, qa_chain)


def get_rag_chain(temperature: float | None = None) -> Runnable:
    """Get RAG chain for standard (non-streaming) usage."""
    return _build_chain(_create_llm(temperature, streaming=False))


def get_streaming_rag_chain(temperature: float | None = None) -> Runnable:
    """Get RAG chain with streaming enabled."""
    return _build_chain(_create_llm(temperature, streaming=True))
