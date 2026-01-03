import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from FlagEmbedding import BGEM3FlagModel
from operator import itemgetter

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
import langchain
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnableLambda
import re
from langchain_core.documents import Document

langchain.debug = True
import numpy as np


def merge_docs(results):
    docs = []
    for r in results.values():
        docs.extend(r)
    return docs


def format_docs(docs):
    authoritative = []
    supporting = []

    for d in docs:
        if d.metadata.get("source_type") == "ryczalt":
            if d.metadata.get("article"):
                authoritative.append(
                    f"[ARTICLE: Art. {d.metadata['article']}].\n{d.page_content}"
                )
            else:
                supporting.append(d.page_content)
        elif d.metadata.get("source_type") == "ryczalt-definition":
            definition_article = "<polish_legal_text>USTAWA o zryczałtowanym podatku dochodowym od niektórych przychodów osiąganych przez osoby fizyczne (ustawa o ryczałcie od przychodów ewidencjonowanych).</polish_legal_text>[ARTICLE: Art. 4].\n"
            authoritative.append(definition_article + d.page_content)
        elif d.metadata.get("source_type") == "faq":
            supporting.append(d.page_content)

    result = ""
    if authoritative:
        result += (
            "AUTHORITATIVE ARTICLES: <polish_legal_text><polish_legal_text>USTAWA o zryczałtowanym podatku dochodowym od niektórych przychodów osiąganych przez osoby fizyczne (ustawa o ryczałcie od przychodów ewidencjonowanych).</polish_legal_text>):\n"
            + "\n\n".join(authoritative)
        )

    if supporting:
        result += (
            "\n\nSUPPORTING MATERIAL (use but clearly inform the user to recheck this information):\n"
            + "\n\n".join(supporting)
        )
    return result


def read_ryczalt_db(article_number):
    with open(f"ryczalt-db/{article_number}.md", "r", encoding="utf-8") as f:
        return f.read()


def get_article_name(line):
    match = re.search(r"Article\s+(\d+[a-zA-Z]*).", line)
    return match.group(1) if match else None


def prioritize_articles(docs):
    articles = [d for d in docs if d.metadata.get("article")]
    return articles if articles else docs


class BGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _normalize(vecs):
        """L2 normalize a numpy array of vectors."""
        vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        return (vecs / norms).tolist()

    def embed_documents(self, texts):
        # Encode documents
        vecs = self.model.encode(texts)["dense_vecs"]
        # Normalize manually
        return self._normalize(vecs)

    def embed_query(self, text):
        # Encode query
        vec = self.model.encode([text])["dense_vecs"]
        # Normalize manually
        return self._normalize(vec)[0]
        
        
def load_lines_as_documents(path: str):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()  

    lines = docs[0].page_content.splitlines()

    documents = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        documents.append(
            Document(
                page_content=line,
                metadata={
                    "source_type": "faq"
                }
            )
        )

    return documents

loader_definition = DirectoryLoader(
    path="ryczalt-definition",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

loader_ryczalt = DirectoryLoader(
    path="ryczalt",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

docs_definition = loader_definition.load()
docs_ryczalt = loader_ryczalt.load()
docs_pkwiu = load_lines_as_documents("pkwiu.txt")

for doc in docs_definition:
    doc.metadata["source_type"] = "ryczalt-definition"
    doc.metadata["article"] = "4" 

for doc in docs_ryczalt:
    doc.metadata["source_type"] = "ryczalt"

docs_with_metadata = []

for doc in docs_ryczalt:  
    article = re.search(r"(\d+[a-zA-Z]*)", doc.metadata["source"])
    docs_with_metadata.append(
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "article": article.group(1)},
        )
    )


os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # dummy but required


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits_definition = text_splitter.split_documents(docs_definition)
splits_ryczalt = text_splitter.split_documents(docs_with_metadata)

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

embeddings_definition = BGEEmbeddings(model)
embeddings_ryczalt = BGEEmbeddings(model)
embeddings_faq = BGEEmbeddings(model)

vectorstore_definition = Chroma(
    collection_name="definition_ryczalt_docs",
    embedding_function=embeddings_definition,
    persist_directory="./chroma/definition",
)

vectorstore_ryczalt = Chroma(
    collection_name="ryzalt_docs",
    embedding_function=embeddings_ryczalt,
    persist_directory="./chroma/ryczalt",
)

vectorstore_faq = Chroma(
    collection_name="faq_docs",
    embedding_function=embeddings_faq,
    persist_directory="./chroma/faq",
)

client_definition = vectorstore_definition._client
client_ryczalt = vectorstore_ryczalt._client
client_faq = vectorstore_faq._client

if vectorstore_definition._collection.count() == 0:
    vectorstore_definition.add_documents(
        [
            doc.model_copy(
                update={"metadata": {**doc.metadata, "source_type": "ryczalt-definition"}}
            )
            for doc in splits_definition
        ]
    )

if vectorstore_ryczalt._collection.count() == 0:
    vectorstore_ryczalt.add_documents(
        [
            doc.model_copy(
                update={"metadata": {**doc.metadata, "source_type": "ryczalt"}}
            )
            for doc in splits_ryczalt
        ]
    )

if vectorstore_faq._collection.count() == 0:
    vectorstore_faq.add_documents(docs_pkwiu)

print(
    f"Dodano {vectorstore_definition._collection.count()} dokumenty do Chroma DB przy uzyciu embedding w z FlagEmbedding."
)

print(
    f"Dodano {vectorstore_ryczalt._collection.count()} dokumenty do Chroma DB przy uzyciu embedding w z FlagEmbedding."
)

print(
    f"Dodano {vectorstore_faq._collection.count()} dokumenty do Chroma DB przy uzyciu embedding w z FlagEmbedding."
)

retriever_definition = vectorstore_definition.as_retriever(
    search_kwargs={
        "k": 1,
        "filter": {"source_type": "ryczalt-definition"},
    }
)

retriever_ryczalt = vectorstore_ryczalt.as_retriever(
    search_kwargs={
        "k": 2,
        "filter": {"source_type": "ryczalt"},
    }
)

retriever_faq = vectorstore_faq.as_retriever(
    search_kwargs={
        "k": 1,
        "filter": {"source_type": "faq"},
    }
)

retriever = RunnableParallel(
    definition=retriever_definition,
    ryczalt=retriever_ryczalt,
    faq=retriever_faq,
) | RunnableLambda(merge_docs)

import time

llm = ChatOllama(model="gemma3:4b", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the question using ONLY the information found in the provided documents.

Rules:
- Information from AUTHORITATIVE ARTICLES has priority and must be treated as true.
- If authoritative articles exist, base your answer primarily on them.
- Specify which article or articles your answer is based on.
Context:
{context}
""",
        ),
        ("user", "{question}"),
    ]
)

# --- 5. Build the RAG Chain ---

rag_chain = (
    {
        "context": (
            itemgetter("question")
            | retriever
            # | RunnableLambda(prioritize_articles)
            | RunnableLambda(format_docs)
        ),
        "question": itemgetter("question"),
    }
    | prompt
    | RunnableLambda(lambda x: print("\nFULL PROMPT:\n", x) or x)
    | llm
    | StrOutputParser()
)

# question = "co jest import uslug jaki przepis o tym mowie"
while True:
    print("\n" + "=" * 50 + "\n")
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower().strip() == "exit":
        break

    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} documents for question.")
    for i, d in enumerate(retrieved_docs):
        print(f"--- Doc {i} ---")
        print(d.page_content)

    start = time.perf_counter()
    response = rag_chain.invoke({"question": question})
    end = time.perf_counter()

    print(f"Question: {question}")
    print(f"Answer: {response}")
    print(f"Time: {end - start}")
