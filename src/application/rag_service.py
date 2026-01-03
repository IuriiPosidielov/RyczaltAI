from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from src.domain.interfaces import VectorStore, LLMProvider, ArticleRepository, DocumentLoader, TextSplitter
from src.domain.models import SourceType, RagDocument

class RagService:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        article_repo: ArticleRepository,
        text_splitter: TextSplitter,
        primary_loader: DocumentLoader,
        definition_loader: DocumentLoader,
        primary_docs_config: dict = None,
        definition_docs_config: dict = None
    ):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.article_repo = article_repo
        self.text_splitter = text_splitter
        self.primary_loader = primary_loader
        self.definition_loader = definition_loader
        
        self.primary_docs_config = primary_docs_config or {
            "collection_name": "vat_docs",
            "source_type": SourceType.VAT.value
        }
        self.definition_docs_config = definition_docs_config or {
            "collection_name": "definition_docs",
            "source_type": SourceType.DEFINITION.value
        }

        self.rag_chain = None
        self.definition_retriever = None
        self.primary_retriever = None

    def initialize_index(self):
        def_col = self.definition_docs_config["collection_name"]
        def_type = self.definition_docs_config["source_type"]
        
        if self.vector_store.get_collection_count(def_col) == 0:
            print(f"Loading and indexing definitions into {def_col}...")
            docs = self.definition_loader.load()
            splits = self.text_splitter.split_documents(docs)
            self.vector_store.add_documents(splits, def_col, def_type)
        
        prim_col = self.primary_docs_config["collection_name"]
        prim_type = self.primary_docs_config["source_type"]

        if self.vector_store.get_collection_count(prim_col) == 0:
            print(f"Loading and indexing primary documents into {prim_col}...")
            docs = self.primary_loader.load()
            splits = self.text_splitter.split_documents(docs)
            self.vector_store.add_documents(splits, prim_col, prim_type)

        self.definition_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 1, "filter": {"source_type": def_type}},
            collection_name=def_col
        )
        self.primary_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 2, "filter": {"source_type": prim_type}},
            collection_name=prim_col
        )
        
        self._build_chain()

    def merge_docs(self, results):
        docs = []
        for r in results.values():
            docs.extend(r)
        return docs

    def prioritize_articles(self, docs):
        articles = [d for d in docs if d.metadata.get("article")]
        return articles if articles else docs

    def format_docs(self, docs):
        authoritative = []
        supporting = []
        source_type_set = set()

        for d in docs:            
            source_type = d.metadata.get("source_type")
            source_type_set.add(source_type)
            article_num = d.metadata.get("article")
            
            if source_type == SourceType.VAT.value:
                if article_num:
                    content = self.article_repo.get_article_content(article_num)
                    authoritative.append(f"[ARTICLE: Art. {article_num}]\n{d.page_content}\n[/ARTICLE: Art. {article_num}]")
                #else:
                #    supporting.append(d.page_content)
            elif source_type == SourceType.DEFINITION.value:
                authoritative.append(f"[ARTICLE: Art. 2]\n{d.page_content}\n[/ARTICLE: Art. 2]")
            elif source_type == SourceType.RYCZALT.value:
                if article_num:
                    content = self.article_repo.get_article_content(article_num)
                    authoritative.append(f"[ARTICLE: Art. {article_num}]\n{d.page_content}\n[/ARTICLE: Art. {article_num}]")
                # else:
                #     supporting.append(d.page_content)
            elif source_type == SourceType.RYCZALT_DEFINITION.value:
                definition_header = "<polish_legal_text>USTAWA o zryczałtowanym podatku dochodowym od niektórych przychodów osiąganych przez osoby fizyczne (ustawa o ryczałcie od przychodów ewidencjonowanych).</polish_legal_text>[ARTICLE: Art. 4].\n"
                authoritative.append(definition_header + d.page_content)

        result = ""
        if authoritative:
            header_text = "AUTHORITATIVE ARTICLES (these are the primary source of truth):"
            if SourceType.VAT.value in source_type_set or SourceType.DEFINITION.value in source_type_set:
                 header_text += "\n<polish_legal_text>Ustawa o podatku od towarów i usług z dnia 11 marca 2004 r. (USTAWA o VAT)</polish_legal_text>):"
            if SourceType.RYCZALT.value in source_type_set or SourceType.RYCZALT_DEFINITION.value in source_type_set:
                 header_text += "\n<polish_legal_text>USTAWA o zryczałtowanym podatku dochodowym od niektórych przychodów osiąganych przez osoby fizyczne (ustawa o ryczałcie od przychodów ewidencjonowanych).</polish_legal_text>):"
            
            result += (
                f"{header_text}\n"
                + "\n\n".join(authoritative)
            )

        if supporting:
            result += (
                "\n\nSUPPORTING MATERIAL (use only if not contradicted by articles):\n"
                + "\n\n".join(supporting)
            )

        return result

    def _build_chain(self):
        retriever_combined = (
            RunnableParallel(
                definition=self.definition_retriever,
                primary=self.primary_retriever,
            )
            | RunnableLambda(self.merge_docs)
        )

        llm = self.llm_provider.get_llm()

#Rules:
#- Information from AUTHORITATIVE ARTICLES has priority and must be treated as true.
#- SUPPORTING MATERIAL can be used.
#- If authoritative articles exist, base your answer primarily on them.
#- Specify which article or articles your answer is based on.



        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
Answer the question using ONLY the information found in the provided articles.
Rules:
- Information from AUTHORITATIVE ARTICLES has priority and must be treated as true.
- If authoritative articles exist, base your answer primarily on them.
- ALWAYS specify which article [ARTICLE: Art. X] or articles [ARTICLE: Art. X] your answer is based on.
Context:
{context}
Please use only the provided articles above to answer the following question.
"""
                ),
                ("user", "{question}"),
            ]
        )

        self.rag_chain = (
            {
                "context": (
                    itemgetter("question")
                    | retriever_combined
                   # | RunnableLambda(self.prioritize_articles)
                    | RunnableLambda(self.format_docs)
                ),
                "question": itemgetter("question"),
            }
            | prompt
           # | RunnableLambda(lambda x: print("\nFULL PROMPT:\n", x) or x)
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        if not self.rag_chain:
            self.initialize_index()
        
        return self.rag_chain.invoke({"question": question})

    def get_retrieved_docs(self, question: str):
        retriever_combined = (
            RunnableParallel(
                definition=self.definition_retriever,
                primary=self.primary_retriever,
            )
            | RunnableLambda(self.merge_docs)
        )
        return retriever_combined.invoke(question)
