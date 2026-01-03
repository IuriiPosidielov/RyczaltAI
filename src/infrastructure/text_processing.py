from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.domain.interfaces import TextSplitter
from src.domain.models import RagDocument
from langchain_core.documents import Document

class LangChainTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents: List[RagDocument]) -> List[RagDocument]:
        # Convert domain RagDocument to LangChain Document
        lc_docs = [
            Document(page_content=d.page_content, metadata=d.metadata)
            for d in documents
        ]
        
        split_docs = self.splitter.split_documents(lc_docs)
        
        # Convert back
        return [
            RagDocument(page_content=d.page_content, metadata=d.metadata)
            for d in split_docs
        ]
