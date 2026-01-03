from config import llm
from state import AgentState
from RAG.ryczalt import ryczalt
from langchain_core.messages import AIMessage
from src.legal_dictionary import legal_dictionary

def handle_ryczalt(state: AgentState):
    last_message = state["messages"][-1].content
    response_content = ryczalt(last_message)
    response = AIMessage(content=response_content)
    print ("category: ", legal_dictionary[state["category"].lower()])
    print ("handle_ryczalt Response: ", response.content)
    return {"messages": [response]}
