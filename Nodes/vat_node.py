from config import llm
from state import AgentState
from RAG.vat import vat
from langchain_core.messages import AIMessage
from src.legal_dictionary import legal_dictionary

def handle_vat(state: AgentState):
    last_message = state["messages"][-1].content
    response_content = vat(last_message)
    response = AIMessage(content=response_content)
    print ("category: ", legal_dictionary[state["category"].lower()])
    print ("handle_vat Response: ", response.content)
    return {"messages": [response]}
