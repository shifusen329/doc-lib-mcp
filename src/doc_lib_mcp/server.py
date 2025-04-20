import asyncio

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# Store notes as a simple key-value dict to demonstrate state management
notes: dict[str, str] = {}

# In-memory chunk store: {source: [chunk, ...]}
chunks_store: dict[str, list[dict]] = {}

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras
import httpx
import os

DB_HOST = os.getenv("HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "docsdb")
DB_USER = os.getenv("DB_USER", "docsdb_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "docsdb_password")

def get_db_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

def ensure_tables():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        source TEXT,
        type TEXT,
        location TEXT,
        content TEXT,
        embedding FLOAT[],
        metadata JSONB
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

# Ensure tables exist at startup
ensure_tables()

# Embedding logic
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text-v1.5-F16")

async def embed_texts(texts):
    url = f"http://{OLLAMA_HOST}:11434/api/embed"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={
            "model": OLLAMA_MODEL,
            "input": texts
        })
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

# Import chunkers
from .chunkers import chunk_markdown, chunk_python, chunk_openapi, chunk_html
import json
import os
import readability # Added import

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
            name="add-note",
            description="Add a new note to the in-memory note store. Arguments: name (note name), content (note content).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the note to add."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the note."
                    },
                },
                "required": ["name", "content"],
            },
        ),
        types.Tool(
            name="ingest-markdown",
            description="Ingest and chunk a markdown (.md) file. Splits the file into chunks by headings or paragraphs and stores them in memory. Argument: path (file path to markdown file).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to the markdown (.md) file to ingest."
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="ingest-python",
            description="Ingest and chunk a Python (.py) file. Splits the file into function/class chunks and stores them in memory. Argument: path (file path to Python file).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to the Python (.py) file to ingest."
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="ingest-openapi",
            description="Ingest and chunk an OpenAPI JSON file. Splits the file into endpoint and schema chunks and stores them in memory. Argument: path (file path to OpenAPI JSON file).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to the OpenAPI JSON file to ingest."
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="ingest-html",
            description="Ingest and chunk an HTML file. Splits the file into chunks by headings or paragraphs and stores them in memory. Argument: path (file path to HTML file).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to the HTML file to ingest."
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="ingest-html-url",
            description="Ingest and chunk HTML content from a URL. If dynamic is true, uses Playwright to fetch rendered HTML (for JS-heavy sites). Arguments: url (string), dynamic (boolean, optional).",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch HTML from."
                    },
                    "dynamic": {
                        "type": "boolean",
                        "description": "Use Playwright to fetch rendered HTML (for JS-heavy sites).",
                        "default": False
                    }
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="search-chunks",
            description="Semantic search over ingested content. Arguments: query (string), top_k (integer, optional, default 3), type (optional), tag (optional, filter by tag in chunk metadata).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return.",
                        "default": 3
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional: Filter results by chunk type (e.g., 'code', 'html', 'markdown')."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional: Filter results by tag in chunk metadata."
                    }
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="delete-source",
            description="Delete all chunks from a given source. Arguments: source (string).",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source file path to delete chunks for."
                    }
                },
                "required": ["source"],
            },
        ),
        types.Tool(
            name="ingest-batch",
            description="Ingest and chunk multiple documentation files (markdown/text, OpenAPI JSON, Python) in batch. Arguments: paths (list of strings).",
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of absolute file paths to ingest."
                    }
                },
                "required": ["paths"],
            },
        ),
        types.Tool(
            name="list-sources",
            description="List all unique sources (file paths) that have been ingested and stored in memory.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="update-chunk-metadata",
            description="Update the metadata field for a chunk by id. Arguments: id (integer), metadata (object).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The id of the chunk to update."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "The metadata dictionary to set (will replace existing metadata)."
                    }
                },
                "required": ["id", "metadata"],
            },
        ),
        types.Tool(
            name="tag-chunks-by-source",
            description="Adds specified tags to the metadata of all chunks associated with a given source (URL or file path). Merges with existing tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "The source identifier (URL or file path) of the chunks to tag."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags to add."
                    }
                },
                "required": ["source", "tags"],
            },
        ),
        types.Tool(
            name="list-notes",
            description="List all currently stored notes and their content.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get-chunks-by-type-and-source",
            description="Retrieve all chunks of a specified type for a provided source id. Arguments: source (string), type (string).",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "The source identifier (file path or URL) to filter by."
                    },
                    "type": {
                        "type": "string",
                        "description": "The chunk type to filter by (e.g., 'code', 'markdown', 'html')."
                    }
                },
                "required": ["source", "type"],
            },
        ),
    ]

SUPERCHUNK_SIZE = 500_000  # 500KB in characters

def split_into_superchunks(text, size=SUPERCHUNK_SIZE):
    """Yield (superchunk_index, superchunk_text) for each superchunk."""
    for i in range(0, len(text), size):
        yield (i // size, text[i:i+size])

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    # Allow tools with no arguments (like list-sources)
    if arguments is None:
        arguments = {}

    if name == "add-note":
        note_name = arguments.get("name")
        content = arguments.get("content")
        if not note_name or not content:
            raise ValueError("Missing name or content")
        notes[note_name] = content
        await server.request_context.session.send_resource_list_changed()
        return [
            types.TextContent(
                type="text",
                text=f"Added note '{note_name}' with content: {content}",
            )
        ]

    elif name == "ingest-markdown":
        path = arguments.get("path")
        if not path or not os.path.isfile(path):
            return [types.TextContent(type="text", text=f"File not found: {path}")]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks = []
        if len(text) > SUPERCHUNK_SIZE:
            for superchunk_index, superchunk in split_into_superchunks(text):
                for chunk in chunk_markdown(superchunk, source=path):
                    chunk["metadata"] = chunk.get("metadata", {})
                    chunk["metadata"]["superchunk_index"] = superchunk_index
                    all_chunks.append(chunk)
        else:
            all_chunks = chunk_markdown(text, source=path)
        embeddings = await embed_texts([chunk["content"] for chunk in all_chunks])
        # Store chunks in the database
        conn = get_db_conn()
        cur = conn.cursor()
        for chunk, emb in zip(all_chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
            )
        conn.commit()
        cur.close()
        conn.close()
        return [
            types.TextContent(
                type="text",
                text=f"Ingested {len(all_chunks)} markdown chunk(s) from {path}",
            )
        ]

    elif name == "ingest-python":
        path = arguments.get("path")
        if not path or not os.path.isfile(path):
            return [types.TextContent(type="text", text=f"File not found: {path}")]
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        all_chunks = []
        if len(code) > SUPERCHUNK_SIZE:
            for superchunk_index, superchunk in split_into_superchunks(code):
                for chunk in chunk_python(superchunk, source=path):
                    chunk["metadata"] = chunk.get("metadata", {})
                    chunk["metadata"]["superchunk_index"] = superchunk_index
                    all_chunks.append(chunk)
        else:
            all_chunks = chunk_python(code, source=path)
        embeddings = await embed_texts([chunk["content"] for chunk in all_chunks])
        # Store chunks in the database
        conn = get_db_conn()
        cur = conn.cursor()
        for chunk, emb in zip(all_chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
            )
        conn.commit()
        cur.close()
        conn.close()
        return [
            types.TextContent(
                type="text",
                text=f"Ingested {len(all_chunks)} Python chunk(s) from {path}",
            )
        ]

    elif name == "ingest-openapi":
        path = arguments.get("path")
        if not path or not os.path.isfile(path):
            return [types.TextContent(type="text", text=f"File not found: {path}")]
        with open(path, "r", encoding="utf-8") as f:
            openapi_text = f.read()
        all_chunks = []
        if len(openapi_text) > SUPERCHUNK_SIZE:
            for superchunk_index, superchunk in split_into_superchunks(openapi_text):
                try:
                    openapi_json = json.loads(superchunk)
                except Exception:
                    continue  # skip invalid JSON superchunks
                for chunk in chunk_openapi(openapi_json, source=path):
                    chunk["metadata"] = chunk.get("metadata", {})
                    chunk["metadata"]["superchunk_index"] = superchunk_index
                    all_chunks.append(chunk)
        else:
            with open(path, "r", encoding="utf-8") as f:
                openapi_json = json.load(f)
            all_chunks = chunk_openapi(openapi_json, source=path)
        embeddings = await embed_texts([chunk["content"] for chunk in all_chunks])
        # Store chunks in the database
        conn = get_db_conn()
        cur = conn.cursor()
        for chunk, emb in zip(all_chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
            )
        conn.commit()
        cur.close()
        conn.close()
        return [
            types.TextContent(
                type="text",
                text=f"Ingested {len(all_chunks)} OpenAPI chunk(s) from {path}",
            )
        ]

    elif name == "ingest-html":
        path = arguments.get("path")
        if not path or not os.path.isfile(path):
            return [types.TextContent(type="text", text=f"File not found: {path}")]
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        all_chunks = []
        if len(html) > SUPERCHUNK_SIZE:
            for superchunk_index, superchunk in split_into_superchunks(html):
                for chunk in chunk_html(superchunk, source=path):
                    chunk["metadata"] = chunk.get("metadata", {})
                    chunk["metadata"]["superchunk_index"] = superchunk_index
                    all_chunks.append(chunk)
        else:
            all_chunks = chunk_html(html, source=path)
        embeddings = await embed_texts([chunk["content"] for chunk in all_chunks])
        # Store chunks in the database
        conn = get_db_conn()
        cur = conn.cursor()
        for chunk, emb in zip(all_chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
            )
        conn.commit()
        cur.close()
        conn.close()
        return [
            types.TextContent(
                type="text",
                text=f"Ingested {len(all_chunks)} HTML chunk(s) from {path}",
            )
        ]

    elif name == "ingest-html-url":
        url = arguments.get("url")
        dynamic = arguments.get("dynamic", False)
        html = None
        if not url:
            return [types.TextContent(type="text", text="Missing URL argument.")]
        if not dynamic:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=30)
                    resp.raise_for_status()
                    html = resp.text
            except Exception as e:
                return [types.TextContent(type="text", text=f"Failed to fetch HTML: {e}")]
        else:
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                return [types.TextContent(type="text", text="Playwright is not installed. Please install with 'pip install playwright' and run 'playwright install chromium'.")]
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, timeout=30000)
                    # Wait for main docs content to load if dynamic retrieval is flagged
                    await page.wait_for_selector('article', timeout=10000)
                    # Extract only the main content node's inner HTML to reduce noise
                    html = await page.eval_on_selector('article', 'el => el.innerHTML')
                    if not html:
                        # Fallback to full page content if selector fails
                        html = await page.content()
                    await browser.close()
            except Exception as e:
                return [types.TextContent(type="text", text=f"Failed to fetch dynamic HTML with Playwright: {e}")]
        if not html:
            return [types.TextContent(type="text", text="No HTML content fetched.")]

        # Clean HTML using readability
        try:
            doc = readability.Document(html)
            cleaned_html = doc.summary()
            title = doc.title() # Optionally capture title
        except Exception as e:
             return [types.TextContent(type="text", text=f"Failed to clean HTML with readability: {e}")]

        if not cleaned_html:
             return [types.TextContent(type="text", text="Readability could not extract main content.")]

        # Chunk the cleaned HTML
        chunks = chunk_html(cleaned_html, source=url)
        if not chunks:
            return [types.TextContent(type="text", text="No content found to ingest from cleaned HTML.")]

        # Add title to metadata of first chunk if available
        if chunks and title:
            if "metadata" not in chunks[0]:
                chunks[0]["metadata"] = {}
            chunks[0]["metadata"]["title"] = title

        embeddings = await embed_texts([c["content"] for c in chunks])
        conn = get_db_conn()
        cur = conn.cursor()
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
            )
        conn.commit()
        cur.close()
        conn.close()
        return [types.TextContent(type="text", text=f"Ingested {len(chunks)} HTML chunk(s) from {url}")]

    elif name == "search-chunks":
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        filter_type = arguments.get("type") # Get optional type filter
        tag = arguments.get("tag") # Get optional tag filter
        if not query:
            return [types.TextContent(type="text", text="Missing query argument.")]

        query_emb = (await embed_texts([query]))[0]
        embedding_str = "[" + ",".join(str(x) for x in query_emb) + "]"
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        sql_query = "SELECT source, type, location, content, metadata, embedding <=> %s AS distance FROM chunks"
        params = [embedding_str]
        where_clauses = []

        if filter_type:
            where_clauses.append("type = %s")
            params.append(filter_type)
        if tag:
            where_clauses.append("metadata->'tags' @> %s")
            params.append(json.dumps([tag]))

        if where_clauses:
            sql_query += " WHERE " + " AND ".join(where_clauses)

        sql_query += " ORDER BY embedding <=> %s LIMIT %s"
        params.extend([embedding_str, top_k])

        cur.execute(sql_query, tuple(params))
        results = cur.fetchall()
        cur.close()
        conn.close()
        if not results:
            return [types.TextContent(type="text", text="No results found.")]
        return [
            types.TextContent(
                type="text",
                text=f"Source: {row['source']}\nType: {row['type']}\nLocation: {row['location']}\nDistance: {row['distance']}\nContent:\n{row['content']}\nMetadata: {json.dumps(row['metadata'])}"
            )
            for row in results
        ]

    elif name == "delete-source":
        source = arguments.get("source")
        if not source:
            return [types.TextContent(type="text", text="Missing source argument.")]
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return [types.TextContent(type="text", text=f"Deleted {deleted} chunk(s) from source: {source}")]

    elif name == "ingest-batch":
        paths = arguments.get("paths")
        if not paths or not isinstance(paths, list):
            return [types.TextContent(type="text", text="Missing or invalid paths argument.")]
        results = []
        for path in paths:
            if not os.path.isfile(path):
                results.append(f"{path}: File not found")
                continue
            if path.endswith(".md"):
                # Reuse ingest-markdown logic
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = chunk_markdown(text, source=path)
                embeddings = await embed_texts([c["content"] for c in chunks])
                conn = get_db_conn()
                cur = conn.cursor()
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                        (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
                    )
                conn.commit()
                cur.close()
                conn.close()
                results.append(f"{path}: Ingested {len(chunks)} markdown chunk(s)")
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    openapi_json = json.load(f)
                chunks = chunk_openapi(openapi_json, source=path)
                embeddings = await embed_texts([c["content"] for c in chunks])
                conn = get_db_conn()
                cur = conn.cursor()
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                        (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
                    )
                conn.commit()
                cur.close()
                conn.close()
                results.append(f"{path}: Ingested {len(chunks)} OpenAPI chunk(s)")
            elif path.endswith(".py"):
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                chunks = chunk_python(code, source=path)
                embeddings = await embed_texts([c["content"] for c in chunks])
                conn = get_db_conn()
                cur = conn.cursor()
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                        (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
                    )
                conn.commit()
                cur.close()
                conn.close()
                results.append(f"{path}: Ingested {len(chunks)} Python chunk(s)")
            elif path.endswith(".html"):
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
                chunks = chunk_html(html, source=path)
                embeddings = await embed_texts([c["content"] for c in chunks])
                conn = get_db_conn()
                cur = conn.cursor()
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(
                        "INSERT INTO chunks (source, type, location, content, embedding, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                        (chunk["source"], chunk["type"], chunk["location"], chunk["content"], emb, json.dumps(chunk.get("metadata", {})))
                    )
                conn.commit()
                cur.close()
                conn.close()
                results.append(f"{path}: Ingested {len(chunks)} HTML chunk(s)")
            else:
                results.append(f"{path}: Unsupported file type")
        return [types.TextContent(type="text", text="\n".join(results))]

    elif name == "list-sources":
        # Query the database for unique sources
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source FROM chunks")
        sources = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        # Wrap each source in TextContent to conform to MCP protocol
        return [
            types.TextContent(type="text", text=source)
            for source in sources
        ]

    elif name == "update-chunk-metadata":
        chunk_id = arguments.get("id")
        metadata = arguments.get("metadata")
        if chunk_id is None or metadata is None:
            return [types.TextContent(type="text", text="Missing id or metadata argument.")]
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE chunks SET metadata = %s WHERE id = %s",
            (json.dumps(metadata), chunk_id)
        )
        updated = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return [types.TextContent(type="text", text=f"Updated metadata for chunk id {chunk_id}. Rows affected: {updated}")]

    elif name == "tag-chunks-by-source":
        source = arguments.get("source")
        tags_to_add = arguments.get("tags")
        if not source or not tags_to_add:
            return [types.TextContent(type="text", text="Missing source or tags argument.")]
        if not isinstance(tags_to_add, list):
             return [types.TextContent(type="text", text="Tags argument must be a list of strings.")]

        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Fetch existing chunks for the source
        cur.execute("SELECT id, metadata FROM chunks WHERE source = %s", (source,))
        chunks_to_update = cur.fetchall()

        updated_count = 0
        for chunk in chunks_to_update:
            chunk_id = chunk['id']
            metadata = chunk['metadata'] if chunk['metadata'] else {}
            existing_tags = set(metadata.get("tags", []))
            new_tags = set(tags_to_add)
            updated_tags = list(existing_tags.union(new_tags))

            if list(existing_tags) != updated_tags: # Only update if tags actually changed
                metadata["tags"] = updated_tags
                cur.execute(
                    "UPDATE chunks SET metadata = %s WHERE id = %s",
                    (json.dumps(metadata), chunk_id)
                )
                updated_count += 1

        conn.commit()
        cur.close()
        conn.close()
        return [types.TextContent(type="text", text=f"Updated tags for {updated_count} chunk(s) from source: {source}")]

    elif name == "list-notes":
        if not notes:
            return [types.TextContent(type="text", text="No notes found.")]
        # Return notes as a JSON string for easy parsing, or format as needed
        notes_content = json.dumps(notes, indent=2)
        return [types.TextContent(type="text", text=notes_content)]

    elif name == "get-chunks-by-type-and-source":
        source = arguments.get("source")
        chunk_type = arguments.get("type")
        if not source or not chunk_type:
            return [types.TextContent(type="text", text="Missing source or type argument.")]
        conn = get_db_conn()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, source, type, location, content, metadata FROM chunks WHERE source = %s AND type = %s",
            (source, chunk_type)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()
        if not results:
            return [types.TextContent(type="text", text=f"No chunks found for source '{source}' with type '{chunk_type}'.")]
        return [
            types.TextContent(
                type="text",
                text=(
                    f"ID: {row['id']}\n"
                    f"Source: {row['source']}\n"
                    f"Type: {row['type']}\n"
                    f"Location: {row['location']}\n"
                    f"Content:\n{row['content']}\n"
                    f"Metadata: {json.dumps(row['metadata'])}"
                )
            )
            for row in results
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
