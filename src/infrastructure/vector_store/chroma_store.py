from typing import List, Dict, Any
from langchain_chroma import Chroma
from src.domain.models import RagDocument
from src.domain.interfaces import VectorStore, EmbeddingsProvider
from langchain_core.documents import Document

class ChromaVectorStore(VectorStore):
    def __init__(self, embeddings_provider: EmbeddingsProvider, persist_directory: str):
        self.embeddings_provider = embeddings_provider
        self.persist_directory = persist_directory
        # We explicitly init per collection in add_documents or as_retriever, 
        # but Chroma API usually wants a collection name at init. 
        # In rag.py, separate stores are created for 'definition' and 'vat'.
        # I will design this class to manage a specific collection.
        self.client_settings = None # Could add settings if needed

    def _get_store(self, collection_name: str) -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings_provider.get_embeddings(),
            persist_directory=f"{self.persist_directory}/{collection_name.split('_')[0]}" # matching rag.py logic (./chroma/definition vs vat)
        )

    def add_documents(self, documents: List[RagDocument], collection_name: str, source_type: str):
        store = self._get_store(collection_name)
        
        # Convert domain RagDocument to LangChain Document
        lc_docs = [
            Document(page_content=d.page_content, metadata=d.metadata)
            for d in documents
        ]
        
        # Check if empty (optimization from rag.py)
        if store._collection.count() == 0:
            store.add_documents(lc_docs)
            print(f"Added {len(lc_docs)} documents to {collection_name}")
        else:
            print(f"Collection {collection_name} already has {store._collection.count()} documents. Skipping add.")

    def as_retriever(self, search_kwargs: Dict[str, Any], collection_name: str):
        store = self._get_store(collection_name)
        return store.as_retriever(search_kwargs=search_kwargs)

    def get_collection_count(self, collection_name: str) -> int:
        store = self._get_store(collection_name)
        return store._collection.count()
