# LangChain and Semantic Kernel Samples

This repository contains a collection of practical examples demonstrating how to build AI-powered applications using both LangChain and Semantic Kernel frameworks. These samples showcase various AI patterns including simple model invocation, tool calling, retrieval-augmented generation (RAG), multi-agent systems, and Model Context Protocol (MCP) integration.

## What's Included

### LangChain Samples

| File | Description |
|------|-------------|
| `1a_simple_model_invoke.py` | Basic example of invoking an Azure OpenAI model and streaming responses |
| `1b_simple_chain.py` | Demonstrates creating and using LangChain chains with prompts and output parsers |
| `2a_manual_tool_calling.py` | Shows how to manually implement tool calling with weather and calculator functions |
| `2b_automatic_tool_calling.py` | Demonstrates automatic tool calling using LangChain's built-in agent capabilities |
| `3_simple_vector_demo.py` | Basic vector store operations and similarity search using in-memory vector storage |
| `4_rag_example.py` | Complete RAG (Retrieval-Augmented Generation) implementation with a cooking assistant |
| `5_multi_agent.py` | Multi-agent system with researcher and writer agents collaborating on tasks |
| `6a_mcp_client.py` | Model Context Protocol client that connects to multiple MCP servers (requires `6b_mcp_server_math.py` to be running first) |
| `6b_mcp_server_math.py` | MCP server providing mathematical calculation tools (start this server before running `6a_mcp_client.py`) |
| `6c_mcp_server_weather.py` | MCP server providing weather information tools |

### Semantic Kernel Samples

| File | Description |
|------|-------------|
| `1_simple_model_invoke.py` | Basic Azure OpenAI model invocation using Semantic Kernel |
| `2_tool_calling.py` | Tool calling implementation with weather and calculator functions |
| `3_simple_vector_demo.py` | Vector search and semantic memory operations |
| `4_openapi_tools.py` | Integration with OpenAPI specifications for external service calls |

## Prerequisites

Before running these samples, ensure you have the following installed:

- **Python 3.13 or higher** - Required for all samples
- **uv** - Modern Python package manager for dependency management
- **Azure OpenAI Service** - Access to Azure OpenAI with a deployed model
- **GPT-4o or similar** - Use GPT-4o or a similarly capable model for best results, especially for tool calling and agent examples

### For VS Code Development (Optional)

If you prefer to run and debug these samples in VS Code, you'll also need:

- **Visual Studio Code** - Latest version recommended
- **Python Extension** - Install the official Python extension by Microsoft
- **Python Debugger Extension** - Install the Python Debugger extension (pylance is also recommended)

### Installing uv

If you don't have uv installed, you can install it using:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd sk-langchain-samples
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```
   This command will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

3. **Create environment configuration:**
   ```bash
   touch .env
   ```

4. **Configure your .env file:**
   Open the `.env` file and add the following environment variables:

   ```env
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
   AZURE_OPENAI_API_VERSION=2024-02-01
   AZURE_OPENAI_API_KEY=your-api-key
   
   # Optional: For embedding models (required for RAG examples)
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name
   ```

## Running the Samples

### Command Line (using uv)

After completing the setup, you can run any sample using uv:

```bash
# Run a LangChain sample
uv run langchain/1a_simple_model_invoke.py

# Run a Semantic Kernel sample
uv run semantickernel/1_simple_model_invoke.py
```

### Visual Studio Code

#### Setting up VS Code

1. **Open the project in VS Code:**
   ```bash
   code sk-langchain-samples
   ```

2. **Configure Python interpreter:**
   - Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from the uv virtual environment (should be automatically detected)

3. **Verify environment setup:**
   - The status bar should show the correct Python interpreter
   - The terminal should use the activated virtual environment

#### Running Samples in VS Code

**Method 1: Run in Terminal**
- Open the integrated terminal (`Ctrl+`` / `Cmd+``)
- Use the same uv commands as above:
  ```bash
  uv run langchain/1a_simple_model_invoke.py
  ```

**Method 2: Run with Python Extension**
- Open any Python file
- Click the "Run Python File" button (▷) in the top-right corner
- Or use `Ctrl+F5` / `Cmd+F5` to run without debugging

**Method 3: Run with Debugging**
- Open any Python file
- Set breakpoints by clicking in the left margin
- Press `F5` or go to Run → Start Debugging
- Choose "Python File" from the debug configuration options
