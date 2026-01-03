from langchain_openai import ChatOpenAI
from src.domain.interfaces import LLMProvider
from typing import Any
import os

class OpenAIProvider(LLMProvider):
    def get_llm(self) -> Any:
        os.environ["OPENAI_API_BASE"] = "http://localhost:8080/v1"
        os.environ["OPENAI_API_KEY"] = "sk-dummy"
        return ChatOpenAI()
