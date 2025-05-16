# doc-lib-mcp MCP server

A Documentation Retrieval Augmented Generation (RAG) MCP server that provides semantic search and question answering capabilities using vector embeddings.

## Components

### Resources

The server implements a simple note storage system with:
- Custom note:// URI scheme for accessing individual notes
- Each note resource has a name, description and text/plain mimetype

### Prompts

The server provides a single prompt:
- summarize-notes: Creates summaries of all stored notes
  - Optional "style" argument to control detail level (brief/detailed)
  - Generates prompt combining all current notes with style preference

### Tools

The server implements three tools:
- ask-question: Ask a question and get a synthesized answer using the RAG agent and contextually relevant information from the embedding database
  - Takes "query" as a required string argument
  - Optional "tags" array to filter or boost relevant chunks
  - Optional "top_n" integer to specify number of results to return
  
- get-embedding: Generate an embedding vector for the given input text using the Ollama API
  - Takes "input" and "model" as required string arguments
  
- get-context: Return the raw context chunks from the embedding database for a given query, without LLM interpretation
  - Takes "query" as a required string argument
  - Optional "tags" array to filter or boost relevant chunks
  - Optional "top_n" integer to specify number of results to return

## Configuration

The server requires the following environment variables (can be set in a .env file):

### Ollama Configuration
- OLLAMA_HOST: Hostname for Ollama API (default: localhost)
- OLLAMA_PORT: Port for Ollama API (default: 11434)
- RAG_AGENT: Ollama model to use for RAG responses (default: llama3)
- OLLAMA_MODEL: Ollama model to use for embeddings (default: nomic-embed-text-v2-moe)

### Database Configuration
- HOST: PostgreSQL database host (default: localhost)
- DB_PORT: PostgreSQL database port (default: 5432)
- DB_NAME: PostgreSQL database name (default: doclibdb)
- DB_USER: PostgreSQL database user (default: doclibdb_user)
- DB_PASSWORD: PostgreSQL database password (default: doclibdb_password)

### Reranker Configuration
- RERANKER_MODEL_PATH: Path to the reranker model (default: /srv/samba/fileshare2/AI/models/bge-reranker-v2-m3)
- RERANKER_USE_FP16: Whether to use FP16 for reranker (default: True)

## Quickstart

### Install

#### Claude Desktop

On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

<details>
  <summary>Development/Unpublished Servers Configuration</summary>
  ```
  "mcpServers": {
    "doc-lib-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/administrator/python-share/doc-lib-mcp",
        "run",
        "doc-lib-mcp"
      ]
    }
  }
  ```
</details>

<details>
  <summary>Published Servers Configuration</summary>
  ```
  "mcpServers": {
    "doc-lib-mcp": {
      "command": "uvx",
      "args": [
        "doc-lib-mcp"
      ]
    }
  }
  ```
</details>

## Development

### Building and Publishing

To prepare the package for distribution:

1. Sync dependencies and update lockfile:
```bash
uv sync
```

2. Build package distributions:
```bash
uv build
```

This will create source and wheel distributions in the `dist/` directory.

3. Publish to PyPI:
```bash
uv publish
```

Note: You'll need to set PyPI credentials via environment variables or command flags:
- Token: `--token` or `UV_PUBLISH_TOKEN`
- Or username/password: `--username`/`UV_PUBLISH_USERNAME` and `--password`/`UV_PUBLISH_PASSWORD`

### Debugging

Since MCP servers run over stdio, debugging can be challenging. For the best debugging
experience, we strongly recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector).


You can launch the MCP Inspector via [`npm`](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) with this command:

```bash
npx @modelcontextprotocol/inspector uv --directory /home/administrator/python-share/doc-lib-mcp run doc-lib-mcp
```


Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.
