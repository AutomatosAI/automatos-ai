#!/usr/bin/env python3
"""
Sync documentation from DeepWiki MCP server into docs/deepwiki/.

Usage:
    python3 scripts/sync_deepwiki_docs.py
    python3 scripts/sync_deepwiki_docs.py --repo AutomatosAI/automatos-ai --output docs/deepwiki --debug

DeepWiki MCP exposes two tools (both take only repoName):
  - read_wiki_structure: returns a text TOC with numbered sections
  - read_wiki_contents: returns ALL pages in one response, separated by "# Page: <title>"

Zero external dependencies — uses only stdlib.
"""

import argparse
import json
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


MCP_ENDPOINT = "https://mcp.deepwiki.com/mcp"
DEFAULT_REPO = "AutomatosAI/automatos-ai"
DEFAULT_OUTPUT = "docs"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 180  # Full content response can be large


# ---------------------------------------------------------------------------
# MCP transport
# ---------------------------------------------------------------------------

def mcp_call(tool_name: str, arguments: dict, debug: bool = False) -> str:
    """Call a DeepWiki MCP tool and return the text content."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        MCP_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )

    if debug:
        print(f"[DEBUG] Calling {tool_name} with {arguments}")

    resp = urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)
    raw = resp.read().decode("utf-8")

    if debug:
        ct = resp.headers.get("Content-Type", "")
        print(f"[DEBUG] Content-Type: {ct}, Size: {len(raw)} bytes")

    # Parse SSE or JSON response
    result = _parse_response(raw)

    # Extract text from MCP content blocks
    return _extract_text(result, debug)


def _parse_response(raw: str) -> dict:
    """Parse either SSE or plain JSON response."""
    # SSE format: lines starting with "data: "
    if raw.lstrip().startswith("event:") or raw.lstrip().startswith("data:"):
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        return json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
        return {}
    return json.loads(raw)


def _extract_text(result: dict, debug: bool = False) -> str:
    """Extract text from MCP result envelope."""
    content = None
    if "result" in result:
        inner = result["result"]
        if isinstance(inner, dict):
            content = inner.get("content")
    if content is None:
        content = result.get("content")

    if not content:
        if debug:
            print(f"[DEBUG] No content in result keys: {list(result.keys())}")
        return ""

    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(item["text"])
        elif isinstance(item, str):
            texts.append(item)
    return "\n".join(texts)


def mcp_call_with_retry(tool_name: str, arguments: dict, debug: bool = False) -> str:
    """Wrap mcp_call with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return mcp_call(tool_name, arguments, debug)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited (429). Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Network error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    return ""


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

def parse_structure(text: str) -> list[dict]:
    """Parse the TOC text into a list of {number, title, parent_section} dicts.

    Input format:
        - 1 Overview
          - 1.1 Key Concepts
          - 1.2 System Architecture
        - 2 Getting Started
          ...
    """
    pages = []
    for line in text.split("\n"):
        # Match lines like "- 1 Overview" or "  - 1.1 Key Concepts"
        m = re.match(r"\s*-\s+(\d+(?:\.\d+)*)\s+(.+)", line)
        if not m:
            continue
        number, title = m.group(1), m.group(2).strip()
        # Determine parent section from number (e.g., "3.2" -> parent "3")
        parts = number.split(".")
        parent = parts[0] if len(parts) > 1 else ""
        pages.append({
            "number": number,
            "title": title,
            "parent": parent,
            "depth": len(parts),
        })
    return pages


def build_section_dirs(pages: list[dict]) -> dict[str, str]:
    """Map section numbers to their parent section titles for directory structure."""
    section_titles = {}
    for p in pages:
        if p["depth"] == 1:
            section_titles[p["number"]] = p["title"]
    return section_titles


# ---------------------------------------------------------------------------
# Content splitting
# ---------------------------------------------------------------------------

def split_pages(full_text: str) -> dict[str, str]:
    """Split the full wiki text by '# Page: <title>' markers.

    Returns {title: content} dict.
    """
    pages = {}
    parts = re.split(r"^# Page: (.+)$", full_text, flags=re.MULTILINE)

    # parts[0] is text before first "# Page:", parts[1] is title, parts[2] is content, etc.
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            pages[title] = content

    return pages


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convert title to filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-") or "untitled"


def write_pages(
    output_dir: Path,
    structure: list[dict],
    content_map: dict[str, str],
    repo: str,
) -> list[dict]:
    """Write all pages to disk using structure for directory layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    section_titles = build_section_dirs(structure)

    # Build a title -> structure entry lookup
    struct_by_title = {p["title"]: p for p in structure}

    written = []

    for title, content in content_map.items():
        entry = struct_by_title.get(title)

        if entry:
            # Use structure to determine path
            if entry["depth"] == 1:
                # Top-level section -> its own directory with index
                subdir = slugify(title)
                filename = "_index.md"
            else:
                # Sub-page -> under parent section directory
                parent_title = section_titles.get(entry["parent"], "")
                subdir = slugify(parent_title) if parent_title else ""
                filename = slugify(title) + ".md"
        else:
            # Page not in structure (shouldn't happen, but handle gracefully)
            subdir = ""
            filename = slugify(title) + ".md"

        target_dir = output_dir / subdir if subdir else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / filename
        filepath.write_text(content, encoding="utf-8")

        written.append({
            "title": title,
            "subdir": subdir,
            "filename": filename,
            "path": str(filepath.relative_to(output_dir)),
        })
        print(f"  {filepath.relative_to(output_dir)}")

    return written


def generate_index(output_dir: Path, structure: list[dict], written: list[dict], repo: str) -> None:
    """Build README.md with hierarchical TOC."""
    written_lookup = {w["title"]: w for w in written}

    lines = [
        "# DeepWiki Documentation",
        "",
        f"Auto-synced from [DeepWiki](https://deepwiki.com/{repo})",
        "",
        f"Last synced: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Table of Contents",
        "",
    ]

    for entry in structure:
        w = written_lookup.get(entry["title"])
        indent = "  " * (entry["depth"] - 1)
        if w:
            lines.append(f"{indent}- [{entry['title']}]({w['path']})")
        else:
            lines.append(f"{indent}- {entry['title']} *(not available)*")

    # Add any pages not in structure
    struct_titles = {e["title"] for e in structure}
    extras = [w for w in written if w["title"] not in struct_titles]
    if extras:
        lines.extend(["", "### Other Pages", ""])
        for w in extras:
            lines.append(f"- [{w['title']}]({w['path']})")

    lines.append("")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Index: {readme_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync DeepWiki docs into repo")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo (default: {DEFAULT_REPO})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--debug", action="store_true", help="Print raw MCP responses")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Step 1: Get structure (TOC)
    print(f"Fetching wiki structure for {args.repo}...")
    structure_text = mcp_call_with_retry("read_wiki_structure", {"repoName": args.repo}, args.debug)
    if not structure_text:
        print("ERROR: Empty structure response. Try --debug.")
        sys.exit(1)

    structure = parse_structure(structure_text)
    print(f"Found {len(structure)} pages in structure")

    if args.debug:
        for s in structure[:10]:
            print(f"  [DEBUG] {s['number']} {s['title']} (depth={s['depth']}, parent={s['parent']})")

    # Step 2: Get all content (single large request)
    print(f"Fetching all wiki content (this may take a moment)...")
    full_content = mcp_call_with_retry("read_wiki_contents", {"repoName": args.repo}, args.debug)
    if not full_content:
        print("ERROR: Empty content response. Try --debug.")
        sys.exit(1)

    print(f"Received {len(full_content):,} bytes of content")

    # Step 3: Split into pages
    content_map = split_pages(full_content)
    print(f"Split into {len(content_map)} pages")

    # Step 4: Write files
    print(f"\nWriting to {output_dir}/:")
    written = write_pages(output_dir, structure, content_map, args.repo)

    # Step 5: Generate index
    generate_index(output_dir, structure, written, args.repo)

    print(f"\nDone! Synced {len(written)} pages to {output_dir}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
