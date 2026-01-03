from typing import List, Any
from langchain_core.embeddings import Embeddings
from FlagEmbedding import BGEM3FlagModel
from src.domain.interfaces import EmbeddingsProvider

class BGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])["dense_vecs"][0].tolist()

class BGEEmbeddingsService(EmbeddingsProvider):
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.embeddings = BGEEmbeddings(self.model)

    def get_embeddings(self) -> Any:
        return self.embeddings
