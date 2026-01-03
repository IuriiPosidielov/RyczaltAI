from abc import ABC, abstractmethod
from typing import List, Any, Dict
from .models import RagDocument

class DocumentLoader(ABC):
    @abstractmethod
    def load(self) -> List[RagDocument]:
        pass

class TextSplitter(ABC):
    @abstractmethod
    def split_documents(self, documents: List[RagDocument]) -> List[RagDocument]:
        pass

class EmbeddingsProvider(ABC):
    @abstractmethod
    def get_embeddings(self) -> Any:
        pass

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[RagDocument], collection_name: str, source_type: str):
        pass
    
    @abstractmethod
    def as_retriever(self, search_kwargs: Dict[str, Any]):
        pass
    
    @abstractmethod
    def get_collection_count(self) -> int:
        pass

class LLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> Any:
        pass

class ArticleRepository(ABC):
    @abstractmethod
    def get_article_content(self, article_number: str) -> str:
        pass
