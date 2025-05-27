# FastAPI RAG Backend

A Retrieval-Augmented Generation (RAG) backend built with FastAPI and ChromaDB.

## Features

- **Multiply AI Clients**: Easily switch between Anthropic, OpenAI, and Google Gemini for LLM generation
- **Vector Database Integration**: Uses ChromaDB for vector storage and retrieval
- **Multiple Embedding Providers**: Support for Voyage AI and OpenAI embeddings
- **Advanced Reranking**: BM25 and Cohere reranking to improve retrieval quality
- **API Authentication**: Preshared key authentication for secure API access
- **Text Chunking**: Automatic document chunking with configurable size and overlap

## Installation

### Prerequisites

- Python 3.13+
- UV (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-api.git
   cd rag-api
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   uv pip install -e .
   ```

3. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   
4. Edit the `.env` file to add your API keys.

### Development Commands
- Run server: `python -m uvicorn src.main:app --reload`
- Format code: `black src/`
- Lint: `uv run ruff check`
- Type check: `mypy src/`
- Run all tests: `pytest`
- Run single test: `pytest path/to/test_file.py::test_function_name`
- Install dependencies: `uv pip install -e .`

## Usage

### Start the API server

```bash
python -m uvicorn src.main:app --reload
```

The API will be available at http://localhost:8000

### API Endpoints

#### Main API
- `GET /api/health`: Check API health status
- `POST /api/documents`: Add documents to the vector database
- `POST /api/documents/chunks`: Add documents after chunking them
- `DELETE /api/documents/{doc_id}`: Delete a document
- `POST /api/search`: Search for documents
- `POST /api/rag`: Generate a response using RAG
- `DELETE /api/collection`: Clear all documents from the collection

#### Configuration API
- `GET /api/config/`: Get all configuration settings
- `GET /api/config/groups`: Get all configuration setting groups
- `GET /api/config/group/{group_name}`: Get settings for a specific group
- `GET /api/config/{key}`: Get a specific configuration setting
- `POST /api/config/`: Create a new configuration setting
- `PUT /api/config/{key}`: Update an existing configuration setting
- `DELETE /api/config/{key}`: Delete a configuration setting
- `POST /api/config/initialize`: Initialize default settings

### Authentication

All API endpoints require authentication using a preshared key. Include the key in the `X-API-Key` header:

```
X-API-Key: your_preshared_key_here
```

## API Examples

### Add Documents

```python
import requests

api_url = "http://localhost:8000/api"
headers = {"X-API-Key": "your_preshared_key_here"}

# Add documents
docs = {
    "documents": [
        {
            "text": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "metadata": {"source": "fastapi_docs", "section": "introduction"}
        },
        {
            "text": "Python is a programming language that lets you work quickly and integrate systems effectively.",
            "metadata": {"source": "python_docs", "section": "about"}
        }
    ]
}

response = requests.post(f"{api_url}/documents", json=docs, headers=headers)
print(response.json())
```

### Search Documents

```python
query = {
    "text": "What is FastAPI?",
    "n_results": 5,
    "rerank_method": "bm25"
}

response = requests.post(f"{api_url}/search", json=query, headers=headers)
print(response.json())
```

### Generate RAG Response

```python
rag_request = {
    "query": "Explain what FastAPI is and why it's used",
    "n_results": 5,
    "temperature": 0.7,
    "llm_provider": "anthropic",
    "rerank_method": "bm25"
}

response = requests.post(f"{api_url}/rag", json=rag_request, headers=headers)
print(response.json())
```

## Configuration

Configuration options are available in `src/core/config.py` and can be set via environment variables in your `.env` file. You can customize:

- API authentication
- LLM providers
- Embedding providers
- Reranking methods
- Vector database settings
- RAG parameters

### Feature Flags

You can enable or disable specific features using environment variables or the configuration API:

```
# Set to False to completely disable embeddings
# This makes the system fall back to keyword search
ENABLE_EMBEDDING=True


# Set to False to disable reranking
# This can improve response time if reranking isn't needed
ENABLE_RERANKING=True
```

When `ENABLE_EMBEDDING` is set to `False`, the system will:
- Skip generating embeddings for documents
- Use ChromaDB's built-in keyword search instead of vector search


When `ENABLE_RERANKING` is set to `False`, the system will:
- Skip the reranking step
- Return results based solely on vector search similarity

### Configuration API Examples

You can use the Configuration API to change settings at runtime:

```python
import requests

api_url = "http://localhost:8000/api"
headers = {"X-API-Key": "your_preshared_key_here"}

# Get all configuration settings
response = requests.get(f"{api_url}/config/", headers=headers)
print(response.json())

# Update a configuration setting
update_data = {
    "value": False,
    "value_type": "bool"
}
response = requests.put(f"{api_url}/config/ENABLE_EMBEDDING", json=update_data, headers=headers)
print(response.json())

# Get settings by group
response = requests.get(f"{api_url}/config/group/features", headers=headers)
print(response.json())
```

The configuration is stored in SQLite and persists across application restarts. The database is created in the same directory as your ChromaDB data.

## License

MIT License