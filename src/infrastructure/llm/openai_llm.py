from langchain_ollama import ChatOllama
from src.domain.interfaces import LLMProvider
from typing import Any

class OpenAILLMProvider(LLMProvider):
    def get_llm(self) -> Any:
        return ChatOllama(model="gemma3:4b", temperature=0)