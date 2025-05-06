import asyncio
import os
import httpx
import requests
import asyncpg
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Ollama API endpoint and model name
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
rag_agent = os.environ.get("RAG_AGENT", "llama3")
embedding_model = os.environ.get("OLLAMA_MODEL", "nomic-embed-text-v2-moe")
generate_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
embed_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# Store notes as a simple key-value dict to demonstrate state management
notes: dict[str, str] = {}

server = Server("doc-lib-mcp")

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available note resources.
    Each note is exposed as a resource with a custom note:// URI scheme.
    """
    return [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific note's content by its URI.
    The note name is extracted from the URI host component.
    """
    if uri.scheme != "note":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    name = uri.path
    if name is not None:
        name = name.lstrip("/")
        return notes[name]
    raise ValueError(f"Note not found: {name}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    The prompt includes all current notes and can be customized via arguments.
    """
    if name != "summarize-notes":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "brief")
    detail_prompt = " Give extensive details." if style == "detailed" else ""

    return types.GetPromptResult(
        description="Summarize the current notes",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                    + "\n".join(
                        f"- {name}: {content}"
                        for name, content in notes.items()
                    ),
                ),
            )
        ],
    )

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="get-context",
            description="Retrieve contextually relevant information from the embedding database and synthesize an answer using the RAG agent. Optionally filter or boost by tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags to filter or boost relevant chunks.",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get-embedding",
            description="Generate an embedding vector for the given input text using the Ollama API.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "model": {"type": "string"}
                },
                "required": ["input", "model"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name == "get-context":
        if not arguments or "query" not in arguments:
            raise ValueError("Missing query argument")
        user_query = arguments["query"]
        tags = arguments.get("tags", None)

        # --- Step 1: Get embedding for the query using Ollama API ---
        # Use requests (sync) for embedding call due to httpx 404 issue
        resp = requests.post(
            embed_url,
            json={"model": embedding_model, "prompt": user_query},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding", [])
        # Convert embedding list to pgvector string format
        embedding = "[" + ",".join(str(x) for x in embedding) + "]"

        # --- Step 2: Query the embedding DB for top-N similar chunks ---
        db_host = os.environ.get("HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "doclibdb")
        db_user = os.environ.get("DB_USER", "doclibdb_user")
        db_password = os.environ.get("DB_PASSWORD", "doclibdb_password")
        table_name = "embeddings"
        chunk_col = "chunk"
        embedding_col = "embedding"
        tags_col = "tags"
        top_n = 5

        conn = await asyncpg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
        )

        # Filtering/boosting by tags
        if tags:
            # Boost: subtract 0.2 from distance if tags overlap
            sql = f"""
                SELECT {chunk_col}, {tags_col}
                FROM {table_name}
                ORDER BY ({embedding_col}::vector <#> $1::vector) - (CASE WHEN {tags_col} && $2 THEN 0.2 ELSE 0 END)
                LIMIT {top_n}
            """
            rows = await conn.fetch(sql, embedding, tags)
        else:
            sql = f"""
                SELECT {chunk_col}, {tags_col}
                FROM {table_name}
                ORDER BY {embedding_col}::vector <#> $1::vector
                LIMIT {top_n}
            """
            rows = await conn.fetch(sql, embedding)
        await conn.close()

        chunks = [row[chunk_col] for row in rows]

        # --- Step 3: Compose prompt for RAG agent ---
        context_str = "\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks))
        prompt = (
            "You are a documentation assistant. Given the following context chunks from a document database, answer the user's question as best as possible.\n\n"
            "If the answer to the user's question is not present in the provided context, respond with 'The answer is not available in the provided documentation.' Do not guess or make up information.\n\n"
            f"User question: {user_query}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Answer:"
        )

        # --- Step 4: Call the RAG agent (Ollama LLM) ---
        async with httpx.AsyncClient() as client:
            gen_resp = await client.post(
                generate_url,
                json={"model": rag_agent, "prompt": prompt, "stream": False},
                timeout=60,
            )
            gen_resp.raise_for_status()
            gen_data = gen_resp.json()
            answer = gen_data.get("response", "")

        return [
            types.TextContent(
                type="text",
                text=answer,
            )
        ]
    elif name == "get-embedding":
        if not arguments or "input" not in arguments or "model" not in arguments:
            raise ValueError("Missing input or model argument")
        input_text = arguments["input"]
        model_name = arguments["model"]
        # Use requests (sync) for embedding call due to httpx 404 issue
        resp = requests.post(
            embed_url,
            json={"model": model_name, "prompt": input_text},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding", [])
        return [
            types.TextContent(
                type="text",
                text=str(embedding),
            )
        ]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="doc-lib-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
