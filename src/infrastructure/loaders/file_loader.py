import re
import os
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from src.domain.models import RagDocument, SourceType
from src.domain.interfaces import DocumentLoader, ArticleRepository

class FileSystemLoader(DocumentLoader):
    def __init__(self, path: str, source_type: SourceType):
        self.path = path
        self.source_type = source_type

    def load(self) -> List[RagDocument]:
        if not os.path.exists(self.path):
            print(f"Warning: Directory {self.path} does not exist.")
            return []
            
        loader = DirectoryLoader(
            path=self.path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()

        if self.source_type == SourceType.DEFINITION:
            for doc in docs:
                doc.metadata["source_type"] = SourceType.DEFINITION.value
                doc.metadata["article"] = "2" 
        elif self.source_type == SourceType.VAT:
            for doc in docs:
                doc.metadata["source_type"] = SourceType.VAT.value
        elif self.source_type == SourceType.RYCZALT:
            for doc in docs:
                doc.metadata["source_type"] = SourceType.RYCZALT.value
        elif self.source_type == SourceType.RYCZALT_DEFINITION:
            for doc in docs:
                doc.metadata["source_type"] = SourceType.RYCZALT_DEFINITION.value
                doc.metadata["article"] = "4" 

        docs_with_metadata = []

        for doc in docs:  # before text splitting
            article = re.search(r"(\d+[a-zA-Z]*)", doc.metadata["source"])
            docs_with_metadata.append(
                RagDocument(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "article": article.group(1)},
                )
            )

        return docs_with_metadata

    #def _get_article_name(self, line: str) -> Optional[str]:
    #    match = re.search(r"Article\s+(\d+[a-zA-Z]*).", line)
    #    return match.group(1) if match else None

class FileArticleRepository(ArticleRepository):
    def __init__(self, base_path: str, file_pattern: str = "vat-{}.md"):
        self.base_path = base_path
        self.file_pattern = file_pattern

    def get_article_content(self, article_number: str) -> str:
        print ("article_number", article_number)
        filename = self.file_pattern.format(article_number)
        path = os.path.join(self.base_path, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"Article {article_number} not found."
