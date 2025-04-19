import re
import ast
from typing import List, Dict, Any

def chunk_markdown(text: str, source: str) -> List[Dict[str, Any]]:
    """
    Split markdown text into chunks by headings (##, #) or paragraphs.
    Each chunk includes content, type, source, location, and metadata.
    """
    heading_regex = re.compile(r'^(#{1,2})\s+(.*)', re.MULTILINE)
    matches = list(heading_regex.finditer(text))
    chunks = []

    if matches:
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            heading = match.group(2).strip()
            content = text[start:end].strip()
            if content:
                chunk = {
                    "content": content,
                    "type": "markdown",
                    "source": source,
                    "location": f"heading:{heading}",
                    "metadata": {"heading": heading}
                }
                chunks.append(chunk)
    else:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for idx, para in enumerate(paragraphs):
            chunk = {
                "content": para,
                "type": "markdown",
                "source": source,
                "location": f"paragraph:{idx+1}",
                "metadata": {}
            }
            chunks.append(chunk)

    return chunks

def chunk_python(code: str, source: str) -> List[Dict[str, Any]]:
    """
    Chunk Python code into functions, classes, and optionally their docstrings.
    Each chunk includes content, type, source, location, and metadata.
    """
    chunks = []
    try:
        tree = ast.parse(code)
    except Exception:
        return [{
            "content": code,
            "type": "python",
            "source": source,
            "location": "full",
            "metadata": {}
        }]

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            docstring = ast.get_docstring(node)
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line)
            lines = code.splitlines()
            func_code = "\n".join(lines[start_line-1:end_line])
            chunk = {
                "content": func_code,
                "type": "python_function",
                "source": source,
                "location": f"function:{name}",
                "metadata": {"name": name, "start_line": start_line, "end_line": end_line}
            }
            chunks.append(chunk)
            if docstring:
                chunks.append({
                    "content": docstring,
                    "type": "python_docstring",
                    "source": source,
                    "location": f"function_docstring:{name}",
                    "metadata": {"name": name, "start_line": start_line, "end_line": end_line}
                })
        elif isinstance(node, ast.ClassDef):
            name = node.name
            docstring = ast.get_docstring(node)
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line)
            lines = code.splitlines()
            class_code = "\n".join(lines[start_line-1:end_line])
            chunk = {
                "content": class_code,
                "type": "python_class",
                "source": source,
                "location": f"class:{name}",
                "metadata": {"name": name, "start_line": start_line, "end_line": end_line}
            }
            chunks.append(chunk)
            if docstring:
                chunks.append({
                    "content": docstring,
                    "type": "python_docstring",
                    "source": source,
                    "location": f"class_docstring:{name}",
                    "metadata": {"name": name, "start_line": start_line, "end_line": end_line}
                })

    if not chunks:
        chunks.append({
            "content": code,
            "type": "python",
            "source": source,
            "location": "full",
            "metadata": {}
        })
    return chunks

def chunk_openapi(openapi_json: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
    """
    Chunk OpenAPI JSON into endpoints (paths/operations) and schema descriptions.
    Each chunk includes content, type, source, location, and metadata.
    """
    chunks = []
    paths = openapi_json.get("paths", {})
    for path, ops in paths.items():
        for method, op in ops.items():
            summary = op.get("summary", "")
            description = op.get("description", "")
            content = f"Path: {path}\nMethod: {method.upper()}\nSummary: {summary}\nDescription: {description}"
            chunk = {
                "content": content.strip(),
                "type": "openapi_operation",
                "source": source,
                "location": f"path:{path}:{method}",
                "metadata": {
                    "path": path,
                    "method": method,
                    "summary": summary,
                    "operationId": op.get("operationId", "")
                }
            }
            chunks.append(chunk)
    components = openapi_json.get("components", {})
    schemas = components.get("schemas", {})
    for name, schema in schemas.items():
        description = schema.get("description", "")
        content = f"Schema: {name}\nDescription: {description}"
        chunk = {
            "content": content.strip(),
            "type": "openapi_schema",
            "source": source,
            "location": f"schema:{name}",
            "metadata": {
                "schema": name,
                "description": description
            }
        }
        chunks.append(chunk)
    if not chunks:
        chunks.append({
            "content": str(openapi_json),
            "type": "openapi",
            "source": source,
            "location": "full",
            "metadata": {}
        })
    return chunks

def chunk_html(text: str, source: str) -> List[Dict[str, Any]]:
    """
    Chunk cleaned HTML text into sections by heading or paragraph, and extract code snippets as dedicated chunks.
    Each chunk includes content, type, source, location, and metadata.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment))):
        comment.extract()

    main = soup.find("main")
    content_root = main if main else soup.body if soup.body else soup

    chunks = []

    # --- Extract code snippets as dedicated chunks ---
    # Find all <pre> tags (which may contain <code>), and all <code> tags not inside <pre>
    code_blocks = []
    for pre in content_root.find_all("pre"):
        code = pre.get_text(separator="\n", strip=True)
        if code:
            code_blocks.append((pre, code))
    for code_tag in content_root.find_all("code"):
        # Skip <code> tags inside <pre>
        if code_tag.find_parent("pre"):
            continue
        code = code_tag.get_text(separator="\n", strip=True)
        if code:
            code_blocks.append((code_tag, code))

    for idx, (tag, code) in enumerate(code_blocks):
        chunks.append({
            "content": code,
            "type": "code",
            "source": source,
            "location": f"code:{idx+1}",
            "metadata": {"code_tag": True}
        })
        tag.decompose()  # Remove code block from soup so it's not included in other chunks

    # --- Chunk the rest of the content as before ---
    headings = content_root.find_all(re.compile("^h[1-3]$"))
    if headings:
        for i, heading in enumerate(headings):
            start = heading
            end = headings[i + 1] if i + 1 < len(headings) else None
            content = []
            for sib in start.next_siblings:
                if sib == end:
                    break
                if getattr(sib, "get_text", None):
                    content.append(sib.get_text(separator=" ", strip=True))
            chunk_text = " ".join(content).strip()
            heading_text = heading.get_text(separator=" ", strip=True)
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "type": "html",
                    "source": source,
                    "location": f"heading:{heading_text}",
                    "metadata": {"heading": heading_text}
                })
    else:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in content_root.find_all("p") if p.get_text(strip=True)]
        for idx, para in enumerate(paragraphs):
            chunks.append({
                "content": para,
                "type": "html",
                "source": source,
                "location": f"paragraph:{idx+1}",
                "metadata": {}
            })

    return chunks
