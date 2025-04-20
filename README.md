# doc-lib-mcp MCP server

A Model Context Protocol (MCP) server for document ingestion, chunking, semantic search, and note management.

## Components

### Resources

- Implements a simple note storage system with:
  - Custom `note://` URI scheme for accessing individual notes
  - Each note resource has a name, description, and `text/plain` mimetype

### Prompts

- Provides a prompt:
  - **summarize-notes**: Creates summaries of all stored notes
    - Optional "style" argument to control detail level (brief/detailed)
    - Generates prompt combining all current notes with style preference

### Tools

The server implements a wide range of tools:

- **add-note**: Add a new note to the in-memory note store
  - Arguments: `name` (string), `content` (string)
- **ingest-markdown**: Ingest and chunk a markdown (.md) file
  - Arguments: `path` (string)
- **ingest-python**: Ingest and chunk a Python (.py) file
  - Arguments: `path` (string)
- **ingest-openapi**: Ingest and chunk an OpenAPI JSON file
  - Arguments: `path` (string)
- **ingest-html**: Ingest and chunk an HTML file
  - Arguments: `path` (string)
- **ingest-html-url**: Ingest and chunk HTML content from a URL (optionally using Playwright for dynamic content)
  - Arguments: `url` (string), `dynamic` (boolean, optional)
- **search-chunks**: Semantic search over ingested content
  - Arguments: 
    - `query` (string): The semantic search query.
    - `top_k` (integer, optional, default 3): Number of top results to return.
    - `type` (string, optional): Filter results by chunk type (e.g., `code`, `html`, `markdown`).
    - `tag` (string, optional): Filter results by tag in chunk metadata.
  - Returns the most relevant chunks for a given query, optionally filtered by type and/or tag.
- **delete-source**: Delete all chunks from a given source
  - Arguments: `source` (string)
- **ingest-batch**: Ingest and chunk multiple documentation files (markdown, OpenAPI JSON, Python) in batch
  - Arguments: `paths` (list of strings)
- **list-sources**: List all unique sources (file paths) that have been ingested and stored in memory
- **update-chunk-metadata**: Update the metadata field for a chunk by id
  - Arguments: `id` (integer), `metadata` (object)
- **tag-chunks-by-source**: Adds specified tags to the metadata of all chunks associated with a given source (URL or file path). Merges with existing tags.
  - Arguments: `source` (string), `tags` (list of strings)
- **list-notes**: List all currently stored notes and their content.

#### Chunking and Code Extraction

- Markdown, Python, OpenAPI, and HTML files are split into logical chunks for efficient retrieval and search.
- The HTML chunker uses the `readability-lxml` library to extract main content first.
- It then extracts block code snippets from `<pre>` tags as dedicated "code" chunks. Inline `<code>` content remains part of the narrative chunks.

#### Semantic Search

- The `search-chunks` tool performs vector-based semantic search over all ingested content, returning the most relevant chunks for a given query.
- Supports optional `type` and `tag` arguments to filter results by chunk type (e.g., `code`, `html`, `markdown`) and/or by tag in chunk metadata, before semantic ranking.
- This enables highly targeted retrieval, such as "all code chunks tagged with 'langfuse' relevant to 'cost and usage'".

#### Metadata Management

- Chunks include a `metadata` field for categorization and tagging.
- The `update-chunk-metadata` tool allows updating metadata for any chunk by its id.
- The `tag-chunks-by-source` tool allows adding tags to all chunks from a specific source in one operation. Tagging merges new tags with existing ones, preserving previous tags.

## Configuration

[TODO: Add configuration details specific to your implementation]

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
        "/home/administrator/python-share/documentation_library/doc-lib-mcp",
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
npx @modelcontextprotocol/inspector uv --directory /home/administrator/python-share/documentation_library/doc-lib-mcp run doc-lib-mcp
```

Upon launching, the Inspector will display a URL that you can access in your browser to begin debugging.
