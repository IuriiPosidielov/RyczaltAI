from config import llm
from state import AgentState
from constants import CATEGORY_VAT, CATEGORY_RYCZALT

def categorize_input(state: AgentState):
    last_message = state["messages"][-1].content
    
    #if last_message.lower() in ["ryczalt", "ryczalt", "pkwiu"]:
    #    return {"category": CATEGORY_RYCZALT}

    #if last_message.lower() in ["vat", "faktur", "invoice"]:
    #    return {"category": CATEGORY_VAT}
    
    prompt = f"""
    Task: Classify the Polish text as '{CATEGORY_VAT}' or '{CATEGORY_RYCZALT}'.
    Definitions (Context):
    VAT: Dotyczy kwestii podatku VAT, wystawiania faktur, stawek VAT oraz JPK.
    Ryczalt: Dotyczy ryczałtu ewidencjonowanego, w tym zagadnień wynikających z ustawy UZPD oraz ustawy UPOF, kody PKWiU.
    Constraints:
    Return ONLY the label ('{CATEGORY_VAT}' or '{CATEGORY_RYCZALT}').
    Do not include punctuation, explanations, or extra words.
    Input: {last_message}
    """
    response = llm.invoke(prompt)
    category = response.content.strip()
    print ("categorize_input Response: ", category)
    
    return {"category": category}
