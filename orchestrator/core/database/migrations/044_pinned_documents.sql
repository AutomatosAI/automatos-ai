-- PRD-157 S5: Document pinning
-- Pin a document to a chat so its content is always injected into that
-- conversation's context (within the retrieval token budget).
--
-- One row per (chat, document); cascade-deletes with either side.

CREATE TABLE IF NOT EXISTS pinned_documents (
    id                 SERIAL PRIMARY KEY,
    chat_id            UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    document_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id       UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    created_at         TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT uq_pinned_chat_document UNIQUE (chat_id, document_id)
);

CREATE INDEX IF NOT EXISTS ix_pinned_documents_chat ON pinned_documents(chat_id);
CREATE INDEX IF NOT EXISTS ix_pinned_documents_workspace ON pinned_documents(workspace_id);
