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
RYCZALT_DOCS_PATH = "ryczalt"
DEF_DOCS_PATH = "ryczalt-definition"

def ryczalt(question=None):
    # Paths
    base_dir = os.getcwd()
    ryczalt_path = os.path.join(base_dir, RYCZALT_DOCS_PATH)
    def_path = os.path.join(base_dir, DEF_DOCS_PATH)
    ryczalt_db_path = os.path.join(base_dir, "ryczalt-db")
    chroma_path = os.path.join(base_dir, CHROMA_PERSIST_DIRECTORY)

    # Components
    ryczalt_loader = FileSystemLoader(ryczalt_path, SourceType.RYCZALT)
    def_loader = FileSystemLoader(def_path, SourceType.RYCZALT_DEFINITION)
    
    article_repo = FileArticleRepository(ryczalt_db_path, file_pattern="{}.md")
    
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
        primary_loader=ryczalt_loader,
        definition_loader=def_loader,
        primary_docs_config={
            "collection_name": "ryczalt_docs", 
            "source_type": SourceType.RYCZALT.value
        },
        definition_docs_config={
            "collection_name": "ryczalt_definition_docs", 
            "source_type": SourceType.RYCZALT_DEFINITION.value
        }
    )


    # 4. Presentation Layer
    # 4. Presentation Layer
    if question:
        rag_service.initialize_index()
        return rag_service.ask(question)
    else:
        console_app = ConsoleInterface(rag_service)
        console_app.run()
