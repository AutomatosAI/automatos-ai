"""
Tree-sitter based multi-language code parser.

Replaces Python ast module and regex-based TypeScript/JavaScript parsing
with tree-sitter for accurate, cross-language symbol and relationship extraction.
Supports 15+ languages via tree-sitter-language-pack.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import tree-sitter-language-pack (preferred)
# Falls back to individual tree-sitter-{lang} packages
_HAS_LANGUAGE_PACK = False
_HAS_TREE_SITTER = False

try:
    from tree_sitter_language_pack import get_language, get_parser
    _HAS_LANGUAGE_PACK = True
    _HAS_TREE_SITTER = True
    logger.info("Using tree-sitter-language-pack for multi-language parsing")
except ImportError:
    try:
        import tree_sitter
        _HAS_TREE_SITTER = True
        logger.info("tree-sitter-language-pack not available, using individual grammars")
    except ImportError:
        logger.warning("tree-sitter not installed — TreeSitterParser will be unavailable")


@dataclass
class ParsedSymbol:
    """A code symbol extracted by tree-sitter."""
    name: str
    symbol_type: str  # 'function', 'class', 'method', 'interface', 'variable'
    file_path: str
    line_number: int
    end_line_number: int
    signature: str
    docstring: Optional[str]
    code_snippet: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedRelationship:
    """A relationship between code symbols extracted by tree-sitter."""
    from_symbol: str  # qualified name or file path
    to_symbol_name: str  # target name (may be unresolved)
    relationship_type: str  # 'calls', 'imports', 'extends', 'implements'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeSitterParseResult:
    """Complete parse result for a single file."""
    symbols: List[ParsedSymbol]
    relationships: List[ParsedRelationship]
    language: str
    errors: List[str] = field(default_factory=list)


class TreeSitterParser:
    """
    Parse source code using tree-sitter grammars.

    Extracts symbols (functions, classes, methods, interfaces) and
    relationships (calls, imports, extends, implements) from 15+ languages.
    """

    SUPPORTED_LANGUAGES: Dict[str, str] = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.jsx': 'javascript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.c': 'c',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.cs': 'c_sharp',
    }

    # Node types that represent symbol definitions (cross-language)
    SYMBOL_NODE_TYPES: Dict[str, str] = {
        # Functions
        'function_definition': 'function',
        'function_declaration': 'function',
        'function_item': 'function',          # Rust
        'method_definition': 'method',
        'method_declaration': 'method',
        'arrow_function': 'function',
        'function_expression': 'function',
        # Classes
        'class_definition': 'class',
        'class_declaration': 'class',
        'class_specifier': 'class',           # C++
        'struct_item': 'class',               # Rust
        'impl_item': 'class',                 # Rust impl blocks
        # Interfaces
        'interface_declaration': 'interface',
        'trait_item': 'interface',             # Rust
        'protocol_declaration': 'interface',   # Swift
    }

    # Node types for relationship extraction
    CALL_NODE_TYPES = {'call_expression', 'call', 'function_call', 'method_invocation'}
    IMPORT_NODE_TYPES = {
        'import_statement', 'import_declaration', 'import_from_statement',
        'use_declaration', 'using_directive', 'include_directive',
        'require_call',
    }
    EXTENDS_NODE_TYPES = {
        'superclass', 'class_heritage', 'extends_clause',
        'superclass_clause', 'base_class_clause',
    }

    def __init__(self):
        self._parser_cache: Dict[str, Any] = {}
        if not _HAS_TREE_SITTER:
            logger.warning("TreeSitterParser initialized but tree-sitter is not available")

    @property
    def available(self) -> bool:
        """Check if tree-sitter is available for parsing."""
        return _HAS_TREE_SITTER

    def get_language_for_extension(self, extension: str) -> Optional[str]:
        """Get language name for a file extension."""
        return self.SUPPORTED_LANGUAGES.get(extension)

    def _get_parser(self, lang_name: str):
        """Get or create a cached parser for a language."""
        if lang_name in self._parser_cache:
            return self._parser_cache[lang_name]

        if _HAS_LANGUAGE_PACK:
            try:
                parser = get_parser(lang_name)
                self._parser_cache[lang_name] = parser
                return parser
            except Exception as e:
                logger.warning(f"Could not get parser for {lang_name}: {e}")
                return None
        else:
            # Fallback to individual tree-sitter-{lang} packages
            return self._get_parser_individual(lang_name)

    def _get_parser_individual(self, lang_name: str):
        """Fallback: try individual tree-sitter-{lang} packages."""
        import tree_sitter

        # Map language names to their package module names
        pkg_map = {
            'python': 'tree_sitter_python',
            'javascript': 'tree_sitter_javascript',
            'typescript': 'tree_sitter_typescript',
        }
        pkg_name = pkg_map.get(lang_name)
        if not pkg_name:
            return None

        try:
            import importlib
            mod = importlib.import_module(pkg_name)
            lang = tree_sitter.Language(mod.language())
            parser = tree_sitter.Parser(lang)
            self._parser_cache[lang_name] = parser
            return parser
        except Exception as e:
            logger.debug(f"Individual parser not available for {lang_name}: {e}")
            return None

    def parse_file(self, file_path: str, content: str, repo_root: str = "") -> TreeSitterParseResult:
        """
        Extract symbols and relationships from a source file.

        Args:
            file_path: Path to the file (used for qualified names)
            content: Source code content
            repo_root: Repository root for relative path computation

        Returns:
            TreeSitterParseResult with symbols, relationships, and language
        """
        ext = Path(file_path).suffix
        lang_name = self.SUPPORTED_LANGUAGES.get(ext)

        if not lang_name:
            return TreeSitterParseResult(
                symbols=[], relationships=[], language='unknown',
                errors=[f"Unsupported extension: {ext}"]
            )

        if not _HAS_TREE_SITTER:
            return TreeSitterParseResult(
                symbols=[], relationships=[], language=lang_name,
                errors=["tree-sitter not installed"]
            )

        parser = self._get_parser(lang_name)
        if not parser:
            return TreeSitterParseResult(
                symbols=[], relationships=[], language=lang_name,
                errors=[f"No parser available for {lang_name}"]
            )

        try:
            tree = parser.parse(content.encode('utf-8'))
        except Exception as e:
            return TreeSitterParseResult(
                symbols=[], relationships=[], language=lang_name,
                errors=[f"Parse error: {e}"]
            )

        # Compute relative path for qualified names
        if repo_root:
            try:
                rel_path = str(Path(file_path).relative_to(repo_root))
            except ValueError:
                rel_path = file_path
        else:
            rel_path = file_path

        content_lines = content.split('\n')

        symbols = self._extract_symbols(tree.root_node, rel_path, content, content_lines, lang_name)
        relationships = self._extract_relationships(tree.root_node, rel_path, content, content_lines, symbols, lang_name)

        return TreeSitterParseResult(
            symbols=symbols,
            relationships=relationships,
            language=lang_name,
        )

    # ── Symbol Extraction ──────────────────────────────────────────────

    def _extract_symbols(
        self, root_node, file_path: str, content: str,
        content_lines: List[str], lang_name: str
    ) -> List[ParsedSymbol]:
        """Walk AST and extract function/class/method/interface definitions."""
        symbols: List[ParsedSymbol] = []
        self._walk_for_symbols(root_node, file_path, content, content_lines, symbols, lang_name, parent_class=None)
        return symbols

    def _walk_for_symbols(
        self, node, file_path: str, content: str,
        content_lines: List[str], symbols: List[ParsedSymbol],
        lang_name: str, parent_class: Optional[str]
    ):
        """Recursively walk tree nodes to find symbol definitions."""
        node_type = node.type

        if node_type in self.SYMBOL_NODE_TYPES:
            symbol = self._node_to_symbol(node, file_path, content, content_lines, lang_name, parent_class)
            if symbol:
                symbols.append(symbol)
                # If this is a class, walk children with parent_class set
                if symbol.symbol_type in ('class', 'interface'):
                    for child in node.children:
                        self._walk_for_symbols(
                            child, file_path, content, content_lines,
                            symbols, lang_name, parent_class=symbol.name
                        )
                    return  # Already walked children

        # Walk children
        for child in node.children:
            self._walk_for_symbols(child, file_path, content, content_lines, symbols, lang_name, parent_class)

    def _node_to_symbol(
        self, node, file_path: str, content: str,
        content_lines: List[str], lang_name: str,
        parent_class: Optional[str]
    ) -> Optional[ParsedSymbol]:
        """Convert a tree-sitter node to a ParsedSymbol."""
        name = self._get_node_name(node, lang_name)
        if not name:
            return None

        base_type = self.SYMBOL_NODE_TYPES[node.type]
        # Reclassify functions inside classes as methods
        if base_type == 'function' and parent_class:
            base_type = 'method'

        line_number = node.start_point[0] + 1  # tree-sitter is 0-indexed
        end_line = node.end_point[0] + 1

        signature = self._get_signature(node, name, base_type, content, lang_name)
        docstring = self._get_docstring(node, content_lines, lang_name)

        # Code snippet: limit to 5000 chars
        snippet_start = node.start_byte
        snippet_end = min(node.end_byte, snippet_start + 5000)
        code_snippet = content[snippet_start:snippet_end]

        qualified_name = f"{file_path}::{parent_class}.{name}" if parent_class and base_type == 'method' else f"{file_path}::{name}"

        metadata: Dict[str, Any] = {'language': lang_name}
        if parent_class:
            metadata['parent_class'] = parent_class

        # Extract base classes for class nodes
        if base_type == 'class':
            bases = self._get_base_classes(node, lang_name)
            if bases:
                metadata['bases'] = bases

        return ParsedSymbol(
            name=name,
            symbol_type=base_type,
            file_path=file_path,
            line_number=line_number,
            end_line_number=end_line,
            signature=signature,
            docstring=docstring,
            code_snippet=code_snippet,
            metadata=metadata,
        )

    def _get_node_name(self, node, lang_name: str) -> Optional[str]:
        """Extract the name from a definition node."""
        # Look for 'name' or 'declarator' child
        for child in node.children:
            if child.type in ('identifier', 'name', 'property_identifier', 'type_identifier'):
                return child.text.decode('utf-8')
            # function_declarator in C/C++
            if child.type == 'function_declarator':
                return self._get_node_name(child, lang_name)
            # For Rust impl blocks, get the type name
            if child.type == 'type_identifier' and node.type == 'impl_item':
                return child.text.decode('utf-8')

        # Arrow functions: look for parent variable_declarator
        if node.type == 'arrow_function' and node.parent:
            if node.parent.type == 'variable_declarator':
                for child in node.parent.children:
                    if child.type in ('identifier', 'name'):
                        return child.text.decode('utf-8')

        return None

    def _get_signature(self, node, name: str, symbol_type: str, content: str, lang_name: str) -> str:
        """Build a human-readable signature."""
        if symbol_type in ('class', 'interface'):
            # Get the first line as signature
            first_line = content[node.start_byte:node.end_byte].split('\n')[0].strip()
            # Truncate long signatures
            return first_line[:200] if len(first_line) > 200 else first_line

        # For functions/methods, extract parameters
        params_node = None
        for child in node.children:
            if child.type in ('parameters', 'formal_parameters', 'parameter_list', 'function_parameters'):
                params_node = child
                break

        if params_node:
            params_text = content[params_node.start_byte:params_node.end_byte]
            # Build signature
            prefix = 'def' if lang_name == 'python' else 'function'
            if node.type == 'arrow_function':
                return f"const {name} = {params_text} =>"
            return f"{prefix} {name}{params_text}"

        # Fallback: first line
        first_line = content[node.start_byte:node.end_byte].split('\n')[0].strip()
        return first_line[:200]

    def _get_docstring(self, node, content_lines: List[str], lang_name: str) -> Optional[str]:
        """Extract docstring or doc comment for a symbol."""
        if lang_name == 'python':
            # Python docstrings: first statement in block containing a string.
            # Grammars with expression_statement as a hidden supertype put the
            # string directly under the block; older ones wrap it.
            for child in node.children:
                if child.type == 'block':
                    for block_child in child.children:
                        string_node = None
                        if block_child.type == 'string':
                            string_node = block_child
                        elif block_child.type == 'expression_statement':
                            for expr_child in block_child.children:
                                if expr_child.type == 'string':
                                    string_node = expr_child
                                    break
                        if string_node is not None:
                            text = string_node.text.decode('utf-8')
                            # Strip triple quotes
                            for q in ('"""', "'''"):
                                if text.startswith(q) and text.endswith(q):
                                    return text[3:-3].strip()
                            return text.strip('"\'').strip()
                        break  # Only check first statement in block
        else:
            # JSDoc / Javadoc: look for comment node immediately before
            line = node.start_point[0]
            if line > 0:
                prev_line = content_lines[line - 1].strip() if line - 1 < len(content_lines) else ''
                if prev_line.startswith('*/'):
                    # Walk back to find /** start
                    doc_lines = []
                    for i in range(line - 1, max(line - 30, -1), -1):
                        l = content_lines[i].strip()
                        doc_lines.insert(0, l)
                        if l.startswith('/**') or l.startswith('/*'):
                            break
                    if doc_lines:
                        doc = '\n'.join(doc_lines)
                        # Clean up comment markers
                        doc = doc.replace('/**', '').replace('*/', '').replace('\n * ', '\n').strip()
                        return doc if doc else None

        return None

    def _get_base_classes(self, node, lang_name: str) -> List[str]:
        """Extract base classes/interfaces from a class definition node."""
        bases = []
        for child in node.children:
            # Python: argument_list after class name
            if child.type == 'argument_list':
                for arg in child.children:
                    if arg.type == 'identifier':
                        name = arg.text.decode('utf-8')
                        if name != 'object':
                            bases.append(name)
            # JS/TS/Java: class_heritage, extends_clause, superclass
            elif child.type in ('class_heritage', 'extends_clause', 'superclass', 'superclass_clause'):
                for sub in child.children:
                    if sub.type in ('identifier', 'type_identifier'):
                        bases.append(sub.text.decode('utf-8'))
            # implements_clause
            elif child.type == 'implements_clause':
                for sub in child.children:
                    if sub.type in ('identifier', 'type_identifier'):
                        bases.append(sub.text.decode('utf-8'))
        return bases

    # ── Relationship Extraction ────────────────────────────────────────

    def _extract_relationships(
        self, root_node, file_path: str, content: str,
        content_lines: List[str], symbols: List[ParsedSymbol], lang_name: str
    ) -> List[ParsedRelationship]:
        """Extract calls, imports, extends, implements relationships from AST."""
        relationships: List[ParsedRelationship] = []

        # Build a symbol name set for context
        symbol_names = {s.name for s in symbols}
        symbol_qualified = {f"{file_path}::{s.name}": s for s in symbols}

        # Extract imports (file-level)
        self._extract_imports(root_node, file_path, content, relationships, lang_name)

        # Extract extends/implements from class definitions
        for sym in symbols:
            if sym.symbol_type in ('class', 'interface'):
                bases = sym.metadata.get('bases', [])
                for base in bases:
                    relationships.append(ParsedRelationship(
                        from_symbol=f"{file_path}::{sym.name}",
                        to_symbol_name=base,
                        relationship_type='extends',
                        metadata={'line': sym.line_number, 'file': file_path},
                    ))

        # Extract function calls within each symbol's scope
        self._extract_calls_from_tree(root_node, file_path, content, relationships, symbols, lang_name)

        return relationships

    def _extract_imports(
        self, node, file_path: str, content: str,
        relationships: List[ParsedRelationship], lang_name: str
    ):
        """Walk tree to find import statements and create import relationships."""
        if node.type in self.IMPORT_NODE_TYPES:
            imported = self._parse_import_node(node, content, lang_name)
            for module_name in imported:
                relationships.append(ParsedRelationship(
                    from_symbol=file_path,
                    to_symbol_name=module_name,
                    relationship_type='imports',
                    metadata={
                        'line': node.start_point[0] + 1,
                        'file': file_path,
                    },
                ))
            return  # Don't recurse into import nodes

        for child in node.children:
            self._extract_imports(child, file_path, content, relationships, lang_name)

    def _parse_import_node(self, node, content: str, lang_name: str) -> List[str]:
        """Parse an import node to extract module/symbol names."""
        imports: List[str] = []
        text = content[node.start_byte:node.end_byte]

        if lang_name == 'python':
            # import foo / from foo import bar
            for child in node.children:
                if child.type == 'dotted_name':
                    imports.append(child.text.decode('utf-8'))
                elif child.type == 'aliased_import':
                    for sub in child.children:
                        if sub.type == 'dotted_name':
                            imports.append(sub.text.decode('utf-8'))
                            break
                elif child.type == 'import_from_statement':
                    imports.extend(self._parse_import_node(child, content, lang_name))
            # If it's "from X import Y", get X
            if node.type == 'import_from_statement':
                for child in node.children:
                    if child.type == 'dotted_name':
                        imports.append(child.text.decode('utf-8'))
                        break

        elif lang_name in ('javascript', 'typescript', 'tsx'):
            # import X from 'Y' / import { X } from 'Y' / require('Y')
            for child in node.children:
                if child.type == 'string' or child.type == 'string_literal':
                    module = child.text.decode('utf-8').strip("'\"")
                    imports.append(module)
                    break

        elif lang_name == 'java':
            # import com.foo.Bar
            for child in node.children:
                if child.type == 'scoped_identifier':
                    imports.append(child.text.decode('utf-8'))
                    break

        elif lang_name == 'go':
            # import "fmt" / import ( "fmt" "os" )
            for child in node.children:
                if child.type == 'import_spec':
                    for sub in child.children:
                        if sub.type == 'interpreted_string_literal':
                            imports.append(sub.text.decode('utf-8').strip('"'))
                elif child.type == 'import_spec_list':
                    for spec in child.children:
                        if spec.type == 'import_spec':
                            for sub in spec.children:
                                if sub.type == 'interpreted_string_literal':
                                    imports.append(sub.text.decode('utf-8').strip('"'))

        elif lang_name == 'rust':
            # use std::collections::HashMap
            for child in node.children:
                if child.type in ('use_as_clause', 'scoped_use_list', 'use_wildcard', 'scoped_identifier', 'identifier'):
                    imports.append(child.text.decode('utf-8'))
                    break

        elif lang_name in ('c', 'cpp'):
            # #include <stdio.h> / #include "myheader.h"
            for child in node.children:
                if child.type in ('system_lib_string', 'string_literal'):
                    imports.append(child.text.decode('utf-8').strip('<>"'))
                    break

        elif lang_name == 'c_sharp':
            # using System.Collections;
            for child in node.children:
                if child.type in ('qualified_name', 'identifier'):
                    imports.append(child.text.decode('utf-8'))
                    break

        else:
            # Generic: try to get the full import text
            if text:
                imports.append(text.strip())

        return imports

    def _extract_calls_from_tree(
        self, node, file_path: str, content: str,
        relationships: List[ParsedRelationship],
        symbols: List[ParsedSymbol], lang_name: str,
        current_scope: Optional[str] = None
    ):
        """Recursively extract function/method call relationships."""
        # Track which scope we're in
        if node.type in self.SYMBOL_NODE_TYPES:
            name = self._get_node_name(node, lang_name)
            if name:
                current_scope = f"{file_path}::{name}"

        # Check if this is a call expression
        if node.type in self.CALL_NODE_TYPES:
            called_name = self._get_call_target(node, content, lang_name)
            if called_name and current_scope:
                relationships.append(ParsedRelationship(
                    from_symbol=current_scope,
                    to_symbol_name=called_name,
                    relationship_type='calls',
                    metadata={
                        'line': node.start_point[0] + 1,
                        'file': file_path,
                    },
                ))

        for child in node.children:
            self._extract_calls_from_tree(
                child, file_path, content, relationships,
                symbols, lang_name, current_scope
            )

    def _get_call_target(self, node, content: str, lang_name: str) -> Optional[str]:
        """Extract the function/method name being called."""
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8')
            if child.type in ('member_expression', 'attribute', 'field_expression'):
                # obj.method() — return the method name
                for sub in child.children:
                    if sub.type in ('property_identifier', 'identifier', 'field_identifier'):
                        return sub.text.decode('utf-8')
                # Fallback: last identifier
                for sub in reversed(child.children):
                    if sub.type in ('identifier', 'property_identifier'):
                        return sub.text.decode('utf-8')
            if child.type == 'scoped_identifier':
                # Rust: Module::function()
                return child.text.decode('utf-8')
        return None
