# Legal AI Assistant

## Overview
This project is an automated legal assistant specialized in Polish tax law, specifically VAT (Value Added Tax) and Ryczałt (Lump-sum tax). It utilizes a Retrieval-Augmented Generation (RAG) architecture powered by LangChain and local LLMs (via Ollama) to answer user questions based on authoritative legal acts and definitions.

The system uses a highly modular graph-based workflow (LangGraph) to process, categorize, and answer queries by routing them to specialized domain handlers.

## Features
- **Intelligent Routing**: Automatically categorizes user queries into VAT or Ryczałt domains using a dedicated Categorizer node.
- **RAG Architecture**: Retrieves precise legal articles and definitions to ground the LLM's responses in actual legal texts.
- **Local LLM Integration**: Configured to work with Ollama, specifically designed for the `gemma3:4b` model (customizable).
- **LangGraph Workflow**: Modular, stateful execution pipeline defined in `graph.py`.
- **Dependency Management**: Uses `uv` for fast and reliable Python package management.

## Prerequisites
- **Python**: 3.10 or higher.
- **Ollama**: Must be installed and running.
  - Pull the required model: `ollama pull gemma3:4b` (or adjust the code to your preferred model).
- **uv**: An extremely fast Python package and project manager.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd LEGAL
    ```

2.  **Install dependencies using `uv`**:
    ```bash
    uv sync
    ```
    This will create a virtual environment (`.venv`) and install all required packages defined in `pyproject.toml`.

## Usage

### Running the Application
The main entry point for the application is `main.py`, which initializes the graph and runs predefined test cases.

To run it using `uv`:
```bash
uv run main.py
```

### Running Specific Modules
You can also invoke other scripts for testing specific components:
- `raggema.py`: Standalone script for testing the RAG pipeline and ChromaDB interactions.

## Project Structure

```
c:/LEGAL
├── Nodes/                  # specialized graph nodes
│   ├── categorizer.py      # routing logic
│   ├── inputtranslator_node.py
│   ├── vat_node.py         # VAT specific handler
│   └── ryczalt_node.py     # Ryczałt specific handler
├── RAG/                    # Retrieval-Augmented Generation logic (per domain)
├── src/                    # Domain Driven Design structure (application, domain, infrastructure)
├── chroma/                 # Persisted Vector Database (ChromaDB)
├── definition/             # Legal definition source files
├── ryczalt/                # Ryczałt legal acts source files
├── ryczalt-db/             # Processed Ryczałt database files
├── vat/                    # VAT legal acts source files
├── graph.py                # LangGraph workflow definition
├── main.py                 # Application entry point
├── raggema.py              # Standalone RAG script/playground
├── config.py             
├── constants.py
└── pyproject.toml          # Project dependencies (uv)
```

## Architecture Flow
1.  **Input**: User query enters the system.
2.  **Input Translator**: Pre-processes the input (if necessary).
3.  **Categorizer**: Analyzes the intent to determine if it's a VAT or Ryczałt question.
4.  **Routing**: The graph routes the state to the appropriate node (`VAT_NODE` or `RYCZALT_NODE`).
5.  **Retrieval & Generation**: The specific node utilizes the RAG pipeline to fetch relevant legal contexts and generates an answer using the LLM.
6.  **Output**: Final answer is returned to the user.
