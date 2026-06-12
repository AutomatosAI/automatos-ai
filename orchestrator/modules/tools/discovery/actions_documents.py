"""Document ActionDefinitions (list, upload, delete, reprocess)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_documents_actions(registry: ActionRegistry) -> None:
    """Register document-related platform actions."""

    registry.register(ActionDefinition(
        name="platform_list_documents",
        description=(
            "List documents uploaded to the workspace knowledge base. "
            "Returns document names, types, sizes, and processing status. "
            "Use when the user asks about their uploaded documents or files."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents to return. Defaults to 50.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["documents", "files", "knowledge"],
        examples=[
            "what documents have I uploaded?",
            "list my files",
            "show knowledge base documents",
        ],
    ))

    # PRD-143 S10: knowledge upload — setup-surface gap-fill.
    registry.register(ActionDefinition(
        name="platform_upload_document",
        description=(
            "Add a text document to the workspace knowledge base from content "
            "Auto already has — markdown (.md), plain text (.txt) or JSON "
            "(.json). The document is stored, chunked and embedded exactly like "
            "a dashboard upload, so agents can retrieve it via RAG. Duplicate "
            "content is detected and not re-uploaded. Use to capture notes, "
            "policies, FAQs or generated knowledge for the workspace."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename including extension — .md, .markdown, .txt or .json.",
                },
                "content": {
                    "type": "string",
                    "description": "The full text content of the document.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional short description of the document.",
                },
            },
            "required": ["filename", "content"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["documents", "knowledge", "upload", "rag", "setup"],
        examples=[
            "add this FAQ to the knowledge base",
            "upload these notes as a document",
            "save this policy as knowledge",
        ],
    ))

    # PRD-157 S2: document-reading tools (Letta pattern). Both read-only and
    # workspace/team-scoped via the centralized retrieval filters.
    registry.register(ActionDefinition(
        name="platform_read_document",
        description=(
            "Read the full text of a knowledge-base document, one page at a time. "
            "Use this to read PAST the short snippet returned by search — pass the "
            "document_id from a search result, then request successive pages to read "
            "the whole document. Each page is a token-budgeted slice; the response "
            "reports total_pages, has_more and next_page so you can keep reading."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "ID of the document to read (from a search result or list_documents).",
                },
                "page": {
                    "type": "integer",
                    "description": "Zero-based page number to read. Defaults to 0 (the first page).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Optional chunk-index to start from; the page containing it is returned.",
                },
            },
            "required": ["document_id"],
        },
        permission_level="read",
        tags=["documents", "read", "knowledge", "rag"],
        examples=[
            "read the rest of that document",
            "show me page 2 of document 12",
            "read document 7 in full",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_grep_documents",
        description=(
            "Search the literal text of knowledge-base documents with a regular "
            "expression and get back the matching passages with their document id "
            "and chunk position. Use for exact-string / pattern lookups (an error "
            "code, a config key, a name) where semantic search is too fuzzy."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "A Python regular expression to match against document text (case-insensitive).",
                },
                "team": {
                    "type": "string",
                    "description": "Optional team to narrow the search within your accessible documents.",
                },
                "document_id": {
                    "type": "integer",
                    "description": "Optional: restrict the search to a single document.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching passages to return. Defaults to 20.",
                },
            },
            "required": ["pattern"],
        },
        permission_level="read",
        tags=["documents", "grep", "search", "knowledge", "rag"],
        examples=[
            "grep the docs for ERR_TIMEOUT",
            "find where the docs mention 'rate limit'",
            "search documents for the exact phrase 'service level agreement'",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_delete_document",
        description=(
            "Delete a document from the knowledge base. Cleans up the S3 file, "
            "vector embeddings, and database record. This is permanent and cannot "
            "be undone. Use when the user explicitly asks to delete a document."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "ID of the document to delete.",
                },
            },
            "required": ["document_id"],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["documents", "delete", "destructive"],
        examples=[
            "delete document 5",
            "remove that document from the knowledge base",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_reprocess_document",
        description=(
            "Re-process a document — regenerate chunks and vector embeddings. "
            "Use when the user asks to re-embed, reindex, or reprocess a document "
            "in the knowledge base."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "integer",
                    "description": "ID of the document to reprocess.",
                },
            },
            "required": ["document_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["documents", "reprocess", "embed", "write"],
        examples=[
            "reprocess document 3",
            "re-embed document 7",
            "reindex that document",
            "regenerate chunks for document 10",
        ],
    ))

    # PRD-167 S6: document-template tools. Let agents discover the workspace's
    # templates and the data each one expects, then fill one via generate_document.
    registry.register(ActionDefinition(
        name="platform_list_templates",
        description=(
            "List the document templates available in this workspace (branded letters, "
            "reports, invoices, etc.). Returns each template's id, name, description, "
            "format and category. Use before generate_document to pick a template, then "
            "call platform_get_template_schema to learn what data it needs."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional filter — pdf, docx or xlsx.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (e.g. 'report', 'invoice', 'letter').",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["documents", "templates", "generate"],
        examples=[
            "what document templates do we have?",
            "list invoice templates",
            "show me the branded report templates",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_template_schema",
        description=(
            "Get the data a document template expects: its variable chips "
            "(user/company/brand/date) and the data.* fields you must supply, plus "
            "sample data. Use this after platform_list_templates and before "
            "generate_document so you fill the template correctly."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "UUID of the template (from platform_list_templates).",
                },
            },
            "required": ["template_id"],
        },
        permission_level="read",
        tags=["documents", "templates", "schema", "generate"],
        examples=[
            "what fields does the Branded Letter template need?",
            "show the schema for that template",
        ],
    ))
