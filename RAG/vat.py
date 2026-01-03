import os
import sys
import langchain

# Ensure src is in path if running as script from root
sys.path.append(os.getcwd())

from src.domain.models import SourceType
from src.infrastructure.loaders.file_loader import FileSystemLoader, FileArticleRepository
from src.infrastructure.text_processing import LangChainTextSplitter
from src.infrastructure.embeddings.bge_embeddings import BGEEmbeddingsService
from src.infrastructure.vector_store.chroma_store import ChromaVectorStore
from src.infrastructure.llm.openai_llm import OpenAILLMProvider
from src.application.rag_service import RagService
from src.presentation.console import ConsoleInterface

CHROMA_PERSIST_DIRECTORY = "./chroma"
VAT_DOCS_PATH = "vat"
DEF_DOCS_PATH = "definition"

def vat(question=None):
    # Paths
    base_dir = os.getcwd()
    vat_path = os.path.join(base_dir, "vat")
    def_path = os.path.join(base_dir, "definition")
    vat_db_path = os.path.join(base_dir, "vat-db")
    chroma_path = os.path.join(base_dir, "chroma")

    # Components
    vat_loader = FileSystemLoader(vat_path, SourceType.VAT)
    def_loader = FileSystemLoader(def_path, SourceType.DEFINITION)
    
    article_repo = FileArticleRepository(vat_db_path)
    
    # Text Splitter
    text_splitter = LangChainTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Embeddings
    # Assuming model weights are available or will be downloaded
    embeddings_service = BGEEmbeddingsService(model_name="BAAI/bge-m3", use_fp16=True)
    
    # Vector Store
    vector_store = ChromaVectorStore(
        embeddings_provider=embeddings_service, 
        persist_directory=chroma_path
    )
    
    # LLM
    llm_provider = OpenAILLMProvider()

    # 3. Application Layer
    rag_service = RagService(
        vector_store=vector_store,
        llm_provider=llm_provider,
        article_repo=article_repo,
        text_splitter=text_splitter,
        primary_loader=vat_loader,
        definition_loader=def_loader
    )

    # 4. Presentation Layer
    # 4. Presentation Layer
    if question:
        rag_service.initialize_index()
        return rag_service.ask(question)
    else:
        console_app = ConsoleInterface(rag_service)
        console_app.run()
