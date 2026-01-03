import sys

class ConsoleInterface:
    def __init__(self, rag_service):
        self.rag_service = rag_service

    def run(self):
        print("Initializing RAG System...")
        self.rag_service.initialize_index()
        print("System Ready.")
        
        while True:
            print("\n" + "="*50 + "\n")
            # Handle Python 2/3 input compatibility if necessary, but 3 is standard
            question = input("Enter your question (or type 'exit' to quit): ")
            if question.lower().strip() == 'exit':
                break

            # Show retrieved docs (mirroring compatibility with original rag.py)
            retrieved_docs = self.rag_service.get_retrieved_docs(question)
            print(f"Retrieved {len(retrieved_docs)} documents for question.")
            for i, d in enumerate(retrieved_docs):
                print(f"--- Doc {i} ---")
                print(d.page_content)

            print("Generating answer...")
            response = self.rag_service.ask(question)

            print(f"Question: {question}")
            print(f"Answer: {response}")
