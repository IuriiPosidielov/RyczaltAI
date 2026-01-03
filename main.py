from langchain_core.messages import HumanMessage
from graph import create_app, run_flow

app = create_app()

# Test 1: A Math Question
print("\n--- TEST 1: Ryczalt ---")
inputs_1 = {"messages": [HumanMessage(content="do ktorej daty trzeba zaplacic ryczałt")]}

run_flow(app, inputs_1)

# Test 2: A Casual Greeting
print("\n--- TEST 2: VAT ---")
inputs_2 = {"messages": [HumanMessage(content="do ktorej daty trzeba zaplacic vat co miesiec")]}

run_flow(app, inputs_2)
