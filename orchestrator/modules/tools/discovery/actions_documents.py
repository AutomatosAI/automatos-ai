"""Document ActionDefinitions (list, delete, reprocess)."""

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
