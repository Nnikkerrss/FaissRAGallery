"""
Нейроассистент для личного кабинета на основе FAISS RAG
"""

from .lk_assistant import LKAssistant
from .assistant_manager import AssistantManager, get_assistant_manager

__version__ = "1.0.0"
__author__ = "AI Assistant Developer"

__all__ = [
    "LKAssistant",
    "AssistantManager",
    "get_assistant_manager"
]