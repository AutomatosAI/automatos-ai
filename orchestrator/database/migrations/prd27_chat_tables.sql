-- PRD-27 Chat Tables Migration
-- ===============================
-- Creates tables for chat sessions, messages, voting, and artifacts
-- Modified to use Integer user_id (matches existing users.id type)

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Chat sessions table
CREATE TABLE IF NOT EXISTS chats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    visibility VARCHAR(20) DEFAULT 'private' CHECK (visibility IN ('private', 'public')),
    last_context JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT unique_user_title UNIQUE(user_id, title)
);

-- Messages table with parts support
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    parts JSONB NOT NULL DEFAULT '[]'::jsonb,
    attachments JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Message voting table
CREATE TABLE IF NOT EXISTS votes (
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    is_upvoted BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (chat_id, message_id)
);

-- Artifacts table (for code snippets, documents, images)
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    kind VARCHAR(20) NOT NULL CHECK (kind IN ('code', 'text', 'image', 'sheet')),
    artifact_metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (id, created_at)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_created_at ON chats(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(chat_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_votes_chat_message ON votes(chat_id, message_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_user_id ON artifacts(user_id);

-- Comments for documentation
COMMENT ON TABLE chats IS 'PRD-27: Chat sessions with users';
COMMENT ON TABLE messages IS 'PRD-27: Individual messages within chat sessions (uses parts array for multimodal content)';
COMMENT ON TABLE votes IS 'PRD-27: User votes (thumbs up/down) on assistant messages';
COMMENT ON TABLE artifacts IS 'PRD-27: Versioned artifacts (code, documents, images) displayed in artifact viewer';

