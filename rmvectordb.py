import os
from langchain_chroma import Chroma
from src.infrastructure.embeddings.bge_embeddings import BGEEmbeddingsService

embeddings_service = BGEEmbeddingsService()
embeddings = embeddings_service.get_embeddings()

vectorstore_definition = Chroma(
    collection_name="definition_docs",
    embedding_function=embeddings,
    persist_directory="./chroma/definition",
)

vectorstore_vat = Chroma(
    collection_name="vat_docs",
    embedding_function=embeddings,
    persist_directory="./chroma/vat",
)

vectorstore_definition.delete(where={"source_type": "definition"})
vectorstore_vat.delete(where={"source_type": "vat"})
