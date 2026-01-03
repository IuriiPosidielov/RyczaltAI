from langgraph.graph import StateGraph, END
from state import AgentState
from Nodes.categorizer import categorize_input
from Nodes.vat_node import handle_vat
from Nodes.ryczalt_node import handle_ryczalt
from Nodes.inputtranslator_node import handle_inputtranslator
from constants import (
    CATEGORIZER_NODE,
    VAT_NODE,
    RYCZALT_NODE,
    CATEGORY_VAT,
    CATEGORY_RYCZALT,
    INPUT_TRANSLATOR_NODE,
)

def routing_logic(state: AgentState):
    if state["category"] == CATEGORY_VAT:
        return VAT_NODE
    return RYCZALT_NODE

def create_app():
    workflow = StateGraph(AgentState)
    workflow.add_node(INPUT_TRANSLATOR_NODE, handle_inputtranslator)
    workflow.add_node(CATEGORIZER_NODE, categorize_input)
    workflow.add_node(VAT_NODE, handle_vat)
    workflow.add_node(RYCZALT_NODE, handle_ryczalt)
    workflow.set_entry_point(INPUT_TRANSLATOR_NODE)

    workflow.add_edge(INPUT_TRANSLATOR_NODE, CATEGORIZER_NODE)
    workflow.add_conditional_edges(
        CATEGORIZER_NODE,
        routing_logic,
        {
            VAT_NODE: VAT_NODE,
            RYCZALT_NODE: RYCZALT_NODE
        }
    )
    workflow.add_edge(VAT_NODE, END)
    workflow.add_edge(RYCZALT_NODE, END)
    return workflow.compile()

def run_flow(app, inputs):
    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"Finished running: {key}")
