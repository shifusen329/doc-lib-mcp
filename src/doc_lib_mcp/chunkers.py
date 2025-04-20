import re
import ast
from typing import List, Dict, Any

def chunk_markdown(text: str, source: str) -> List[Dict[str, Any]]:
    """
    Split markdown text into chunks by headings (##, #) or paragraphs,
    and also extract code blocks as dedicated code chunks (redundantly).
    Each chunk includes content, type, source, location, and metadata.
    """
    import re

    # 1. Chunk as usual (headings/paragraphs), leaving code blocks in place
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

    # 2. Extract all fenced code blocks as dedicated code chunks (only triple backticks)
    # Only match code blocks with both opening and closing triple backticks
    code_block_re = re.compile(
        r"^(```+)\s*([^\n`]*)\n(.*?)(?=\n\1(\s|$))", re.DOTALL | re.MULTILINE
    )
    code_blocks = []
    for match in code_block_re.finditer(text):
        fence, lang, code, *_ = match.groups()
        code_content = code.rstrip("\n")
        # Add as code chunk if there is any content between triple backticks
        if code_content:
            meta = {"code_tag": True}
            lang = lang.strip() if lang else None
            if lang:
                meta["language"] = lang
            # If the language looks like a filename (e.g., "main.py", ".env"), add as filename
            if lang and (lang.endswith(".py") or lang.startswith(".")):
                meta["filename"] = lang
            code_blocks.append({
                "content": code_content,
                "type": "code",
                "source": source,
                "location": f"code:{len(code_blocks)+1}",
                "metadata": meta
            })

    chunks.extend(code_blocks)
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

    # --- Extract code blocks from <div class="nextra-code"> wrappers (Nextra/modern docs) ---
    nextra_code_blocks = []
    for code_div in content_root.find_all("div", class_=lambda c: c and "nextra-code" in c):
        pre = code_div.find("pre")
        code = pre.get_text(separator="\n", strip=True) if pre else None
        lang = None
        filename = None
        # Try to extract language from <code> or <pre> class
        if pre and pre.has_attr("class"):
            for cls in pre["class"]:
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
        # Look for filename in a child or sibling div
        filename_div = code_div.find("div", class_=lambda c: c and ("filename" in c or "file" in c))
        if filename_div:
            filename = filename_div.get_text(strip=True)
        if code:
            nextra_code_blocks.append((code_div, code, lang, filename))
        # Remove code block from soup so it's not included in narrative chunks
        code_div.decompose()

    for idx, (tag, code, lang, filename) in enumerate(nextra_code_blocks):
        meta = {"code_tag": True, "framework": "nextra"}
        if lang:
            meta["language"] = lang
        if filename:
            meta["filename"] = filename
        chunks.append({
            "content": code,
            "type": "code",
            "source": source,
            "location": f"nextra-code:{idx+1}",
            "metadata": meta
        })

    # --- Extract block code snippets (<pre>) as dedicated chunks ---
    # Find all <pre> tags. Inline <code> tags will remain part of the narrative chunks.
    pre_blocks = []
    for pre in content_root.find_all("pre"):
        code = pre.get_text(separator="\n", strip=True)
        # Try to extract language from class (e.g., class="language-python")
        lang = None
        filename = None
        if pre.has_attr("class"):
            for cls in pre["class"]:
                if cls.startswith("language-"):
                    lang = cls.replace("language-", "")
        # Look for a filename label in a preceding sibling or parent
        label = None
        parent = pre.parent
        if parent and parent.name == "div" and parent.has_attr("class"):
            for cls in parent["class"]:
                if "filename" in cls or "file" in cls:
                    label = parent.get_text(strip=True)
        # Some doc frameworks use a <div> with a filename label just before the <pre>
        prev = pre.find_previous_sibling()
        if prev and prev.name == "div" and ("filename" in prev.get("class", []) or "file" in prev.get("class", [])):
            filename = prev.get_text(strip=True)
        if code:
            pre_blocks.append((pre, code, lang, filename or label))
        # Remove code block from soup so it's not included in narrative chunks
        pre.decompose()

    for idx, (tag, code, lang, filename) in enumerate(pre_blocks):
        meta = {"code_tag": True}
        if lang:
            meta["language"] = lang
        if filename:
            meta["filename"] = filename
        chunks.append({
            "content": code,
            "type": "code",
            "source": source,
            "location": f"code:{idx+1}",
            "metadata": meta
        })

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
