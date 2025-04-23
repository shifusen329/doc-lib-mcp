import re
import ast
from typing import List, Dict, Any
import json
import pypandoc # Use pypandoc
import subprocess # Needed if pypandoc fails or for direct calls

# Helper function to extract text from Pandoc AST inline elements
def stringify_inline(inlines):
    result = []
    for inline in inlines:
        if inline['t'] == 'Str':
            result.append(inline['c'])
        elif inline['t'] == 'Code':
            result.append(f"`{inline['c'][1]}`")
        elif inline['t'] == 'Emph':
            result.append(f"*{stringify_inline(inline['c'])}*")
        elif inline['t'] == 'Strong':
            result.append(f"**{stringify_inline(inline['c'])}**")
        elif inline['t'] == 'Strikeout':
            result.append(f"~~{stringify_inline(inline['c'])}~~")
        elif inline['t'] == 'Link':
            link_text = stringify_inline(inline['c'][1])
            url = inline['c'][2][0]
            title = inline['c'][2][1]
            title_str = f' "{title}"' if title else ""
            result.append(f"[{link_text}]({url}{title_str})")
        elif inline['t'] == 'Image':
            alt_text = stringify_inline(inline['c'][1])
            url = inline['c'][2][0]
            title = inline['c'][2][1]
            title_str = f' "{title}"' if title else ""
            result.append(f"![{alt_text}]({url}{title_str})")
        elif inline['t'] == 'Space':
            result.append(' ')
        elif inline['t'] == 'SoftBreak':
            result.append('\n')
        elif inline['t'] == 'LineBreak':
            result.append('  \n')
        # Add other inline types as needed (RawInline, Math, etc.)
    return "".join(result)

# Heading and Code Block based chunk_markdown
def chunk_markdown(text: str, source: str) -> List[Dict[str, Any]]:
    """
    Split markdown text into chunks based on headings (H1, H2, H3) and fenced code blocks.
    Includes content before the first heading/code block.
    """
    chunks = []
    lines = text.splitlines()
    current_markdown_chunk_lines = []
    current_heading = None
    current_heading_level = None
    chunk_idx = 0

    # Regex to find headings (H1, H2, H3)
    heading_regex = re.compile(r"^(#+)\s+(.*)$")
    # Regex for fenced code block start, capturing language
    fenced_code_start_regex = re.compile(r"^```+(\S*)$")
    # Regex for fenced code block end
    fenced_code_end_regex = re.compile(r"^```+\s*$")


    in_fenced_code_block = False
    fenced_code_lines = []
    fenced_code_lang = None

    def add_current_markdown_chunk():
        nonlocal chunks, current_markdown_chunk_lines, current_heading, current_heading_level, chunk_idx
        if current_markdown_chunk_lines:
            content = "\n".join(current_markdown_chunk_lines).strip()
            if content:
                metadata = {}
                location = f"chunk:{chunk_idx}"
                if current_heading:
                    metadata['heading'] = current_heading
                    metadata['level'] = current_heading_level
                    location = f"heading:{current_heading_level}:{current_heading}"

                chunks.append({
                    "content": content,
                    "type": "markdown",
                    "source": source,
                    "location": location,
                    "metadata": metadata
                })
                chunk_idx += 1
            current_markdown_chunk_lines = []
            current_heading = None # Reset heading after adding chunk
            current_heading_level = None # Reset level after adding chunk

    for line in lines:
        # Check for fenced code block start
        fenced_code_start_match = fenced_code_start_regex.match(line)
        if fenced_code_start_match and not in_fenced_code_block:
            add_current_markdown_chunk() # Add any preceding text as a markdown chunk
            in_fenced_code_block = True
            fenced_code_lines.append(line)
            fenced_code_lang = fenced_code_start_match.group(1) if fenced_code_start_match.group(1) else None
            continue

        # Check for fenced code block end
        fenced_code_end_match = fenced_code_end_regex.match(line)
        if fenced_code_end_match and in_fenced_code_block:
            fenced_code_lines.append(line)
            in_fenced_code_block = False
            # Add the code block as a separate chunk
            # Ensure we have at least a start and end fence before processing
            if len(fenced_code_lines) >= 2:
                # Include the fences in the chunk content for clarity
                full_code_content = "\n".join(fenced_code_lines).strip()
                if full_code_content:
                    metadata = {"code_tag": True}
                    if fenced_code_lang:
                        metadata["language"] = fenced_code_lang
                    chunks.append({
                        "content": full_code_content,
                        "type": "code",
                        "source": source,
                        "location": f"code:{chunk_idx}",
                        "metadata": metadata
                    })
                    chunk_idx += 1
            fenced_code_lines = []
            fenced_code_lang = None
            continue

        # If in a fenced code block, just append the line
        if in_fenced_code_block:
            fenced_code_lines.append(line)
            continue

        # Check for headings if not in a code block
        heading_match = heading_regex.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            if level <= 3: # Only chunk by H1, H2, H3
                add_current_markdown_chunk() # Add content before this heading as a markdown chunk
                current_heading = heading_match.group(2).strip()
                current_heading_level = level
                current_markdown_chunk_lines.append(line) # Include the heading line in the chunk
                continue

        # If not in a code block and not a heading, add to current markdown chunk lines
        current_markdown_chunk_lines.append(line)


    # Add the last markdown chunk if any lines remain and we are not in a code block
    if not in_fenced_code_block:
        add_current_markdown_chunk()
    # If we end while still in a code block, the last lines are part of an unclosed block - handle as markdown?
    elif fenced_code_lines:
         chunks.append({
             "content": "\n".join(fenced_code_lines).strip(),
             "type": "markdown", # Treat as markdown if block is unclosed
             "source": source,
             "location": f"unclosed_code_block:{chunk_idx}",
             "metadata": {"code_tag": True, "unclosed": True}
         })
         chunk_idx += 1


    # Fallback if no chunks were created (e.g., empty file or only H4+ headings/malformed content)
    if not chunks and text.strip():
         chunks.append({
             "content": text, "type": "markdown", "source": source,
             "location": "full_fallback", "metadata": {"reason": "No chunks generated by heading/code logic"}
         })


    return chunks


# --- UNCHANGED FUNCTIONS BELOW ---

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
