from config import llm
from state import AgentState
from RAG.inputtranslator import inputtranslator
from langchain_core.messages import AIMessage

def handle_inputtranslator(state: AgentState):
    last_message = state["messages"][-1].content
    response_content = inputtranslator(last_message)
    response = AIMessage(content=response_content)
    print ("handle_inputtranslator Response: ", response.content)
    return {"messages": [response]}