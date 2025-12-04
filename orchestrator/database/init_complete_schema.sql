-- ================================================================
-- AUTOMATOS AI PLATFORM - COMPLETE DATABASE SCHEMA
-- ================================================================
-- Single source of truth for database initialization
-- Generated: October 2025
-- PostgreSQL 15+ with pgvector extension
-- ================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ================================================================
-- CORE TABLES
-- ================================================================

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- CHAT TABLES (PRD-27)
-- ================================================================

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

-- Indexes for chat tables
CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chats_created_at ON chats(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(chat_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_votes_chat_message ON votes(chat_id, message_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_user_id ON artifacts(user_id);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    agent_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    configuration JSON,
    performance_metrics JSON,
    tags JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    priority_level VARCHAR(50) DEFAULT 'medium',
    max_concurrent_tasks INTEGER DEFAULT 5,
    auto_start BOOLEAN DEFAULT FALSE,
    quality_score FLOAT,
    emergence_score FLOAT,
    performance FLOAT,
    reliability FLOAT,
    readiness FLOAT,
    coherence FLOAT,
    efficiency FLOAT,
    eci FLOAT,
    validity FLOAT,
    discriminatory_power FLOAT,
    model_config JSONB DEFAULT '{"provider": "openai", "model_id": "gpt-4", "temperature": 0.7}'::jsonb,
    model_usage_stats JSONB DEFAULT '{"total_tokens": 0, "total_cost": 0.0}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_model_config ON agents USING GIN (model_config);

-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    skill_type VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    implementation TEXT,
    parameters JSON,
    performance_data JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    -- PRD-22: Enhanced fields for Git-backed skills and progressive disclosure
    prompt_template TEXT,
    skill_version VARCHAR(20),
    skill_source VARCHAR(50),
    git_repo_url TEXT,
    git_commit_sha VARCHAR(40),
    git_branch VARCHAR(100),
    filesystem_path TEXT,
    tags JSONB,
    skill_metadata JSONB,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    -- PRD-22: Tools schema for executable tool definitions
    tools_schema JSONB DEFAULT NULL
);

-- Agent-Skills many-to-many
CREATE TABLE IF NOT EXISTS agent_skills (
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    PRIMARY KEY (agent_id, skill_id)
);

-- PRD-22: Skill Files table for progressive disclosure
CREATE TABLE IF NOT EXISTS skill_files (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    content_summary TEXT,
    file_size_bytes INTEGER,
    estimated_tokens INTEGER,
    load_level INTEGER NOT NULL DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skill_files_skill_id ON skill_files(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_files_type ON skill_files(file_type);
CREATE INDEX IF NOT EXISTS idx_skill_files_level ON skill_files(load_level);

-- PRD-22: Skill Sources table for Git repository tracking
CREATE TABLE IF NOT EXISTS skill_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) NOT NULL UNIQUE,
    source_type VARCHAR(50) NOT NULL,
    git_url TEXT,
    git_branch VARCHAR(100) DEFAULT 'main',
    git_commit_sha VARCHAR(40),
    local_cache_path TEXT NOT NULL,
    auto_update BOOLEAN DEFAULT FALSE,
    update_frequency_hours INTEGER DEFAULT 24,
    skills_discovered INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_sync_status VARCHAR(50),
    last_sync_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    created_by VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_skill_sources_type ON skill_sources(source_type);
CREATE INDEX IF NOT EXISTS idx_skill_sources_status ON skill_sources(status);
CREATE INDEX IF NOT EXISTS idx_skill_sources_name ON skill_sources(source_name);

-- PRD-22: Skill Versions table
CREATE TABLE IF NOT EXISTS skill_versions (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    git_commit_sha VARCHAR(40),
    changelog TEXT,
    is_current BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    created_by VARCHAR(255),
    UNIQUE(skill_id, version)
);

CREATE INDEX IF NOT EXISTS idx_skill_versions_skill_id ON skill_versions(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_versions_current ON skill_versions(is_current);

-- PRD-22: Skill Audit Log table
CREATE TABLE IF NOT EXISTS skill_audit_log (
    id SERIAL PRIMARY KEY,
    skill_id INTEGER REFERENCES skills(id) ON DELETE SET NULL,
    source_id INTEGER REFERENCES skill_sources(id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,
    action_details JSONB,
    user_id VARCHAR(255),
    ip_address VARCHAR(50),
    user_agent TEXT,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skill_audit_action ON skill_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_skill_audit_created ON skill_audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_skill_audit_skill ON skill_audit_log(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_audit_source ON skill_audit_log(source_id);

-- Additional indexes for skills
CREATE INDEX IF NOT EXISTS idx_skills_source_active ON skills(skill_source, is_active);
CREATE INDEX IF NOT EXISTS idx_skills_tags_gin ON skills USING GIN (tags jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_skills_tools_schema ON skills USING GIN (tools_schema);

-- Patterns table
CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSON,
    usage_count INTEGER DEFAULT 0,
    effectiveness_score FLOAT DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- ================================================================
-- WORKFLOW TABLES
-- ================================================================

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    goal TEXT,
    context TEXT,
    workflow_definition JSON,
    status VARCHAR(50) DEFAULT 'draft',
    owner VARCHAR(255),
    tags JSONB DEFAULT '[]'::jsonb,
    default_policy_id VARCHAR(128),
    last_execution JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    priority VARCHAR(50),
    expected_duration INTEGER,
    complexity_score FLOAT,
    success_rate FLOAT
);

CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_tags ON workflows USING GIN (tags);

-- Workflow-Agents many-to-many
CREATE TABLE IF NOT EXISTS workflow_agents (
    workflow_id INTEGER REFERENCES workflows(id) ON DELETE CASCADE,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    PRIMARY KEY (workflow_id, agent_id)
);

-- Workflow Executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id SERIAL PRIMARY KEY,
    workflow_id INTEGER REFERENCES workflows(id),
    agent_id INTEGER REFERENCES agents(id),
    status VARCHAR(50) DEFAULT 'pending',
    input_data JSON,
    output_data JSON,
    execution_log TEXT,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_message TEXT,
    execution_metadata JSON DEFAULT '{}'::json,
    models_used JSONB DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_models ON workflow_executions USING GIN (models_used);

-- Workflow Templates table
CREATE TABLE IF NOT EXISTS workflow_templates (
    id SERIAL PRIMARY KEY,
    template_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    tags JSONB DEFAULT '[]'::jsonb,
    difficulty VARCHAR(50) DEFAULT 'intermediate',
    template_definition JSONB NOT NULL,
    recommended_agents JSONB DEFAULT '[]'::jsonb,
    estimated_time VARCHAR(50),
    required_tools JSONB DEFAULT '[]'::jsonb,
    use_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    popularity INTEGER DEFAULT 0,
    average_rating FLOAT DEFAULT 0.0,
    is_public BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    is_system BOOLEAN DEFAULT FALSE,
    icon VARCHAR(50),
    preview_image VARCHAR(500),
    documentation_url VARCHAR(500),
    version VARCHAR(50) DEFAULT '1.0',
    changelog JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW() NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    last_used_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_workflow_templates_category ON workflow_templates(category);
CREATE INDEX IF NOT EXISTS idx_workflow_templates_template_id ON workflow_templates(template_id);

-- ================================================================
-- DOCUMENT & KNOWLEDGE TABLES
-- ================================================================

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_type VARCHAR(100),
    file_size INTEGER,
    file_path VARCHAR(500),
    content_hash VARCHAR(255),
    status VARCHAR(50) DEFAULT 'uploaded',
    chunk_count INTEGER DEFAULT 0,
    tags JSON,
    description TEXT,
    doc_metadata JSON,
    upload_date TIMESTAMP DEFAULT NOW(),
    processed_date TIMESTAMP,
    created_by VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Document Chunks table (with pgvector)
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);

-- Document Usage Tracking
CREATE TABLE IF NOT EXISTS document_usage (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    query TEXT,
    results_count INTEGER DEFAULT 0,
    execution_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(255),
    session_id VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_document_usage_event_type ON document_usage(event_type);
CREATE INDEX IF NOT EXISTS idx_document_usage_timestamp ON document_usage(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_document_usage_metadata ON document_usage USING GIN (metadata);

-- ================================================================
-- MULTIMODAL KNOWLEDGE BASE (PRD-19, Migration 006)
-- ================================================================

-- Knowledge Base Types Registry
CREATE TABLE IF NOT EXISTS kb_types (
    id SERIAL PRIMARY KEY,
    type_name VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    processor_class VARCHAR(255),
    storage_strategy VARCHAR(100),
    supports_embedding BOOLEAN DEFAULT true,
    supports_search BOOLEAN DEFAULT true,
    supports_relationships BOOLEAN DEFAULT false,
    enabled BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_types_name ON kb_types(type_name);
CREATE INDEX IF NOT EXISTS idx_kb_types_enabled ON kb_types(enabled);

-- Unified Knowledge Items Table (polymorphic storage)
CREATE TABLE IF NOT EXISTS knowledge_items (
    id SERIAL PRIMARY KEY,
    kb_type_id INTEGER REFERENCES kb_types(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    source_type VARCHAR(100),
    source_id VARCHAR(255),
    title VARCHAR(500),
    content TEXT NOT NULL,
    summary TEXT,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    quality_score FLOAT DEFAULT 0.0,
    importance_score FLOAT DEFAULT 0.0,
    complexity_score FLOAT DEFAULT 0.0,
    confidence_score FLOAT DEFAULT 1.0,
    visibility VARCHAR(50) DEFAULT 'system',
    owner_id VARCHAR(255),
    permissions JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP,
    indexed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_items_type ON knowledge_items(kb_type_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_parent ON knowledge_items(parent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_source ON knowledge_items(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_status ON knowledge_items(status);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_owner ON knowledge_items(owner_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_quality ON knowledge_items(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_importance ON knowledge_items(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_embedding ON knowledge_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_metadata ON knowledge_items USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_content_fts ON knowledge_items USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_knowledge_items_title_fts ON knowledge_items USING GIN (to_tsvector('english', title));

-- Multimodal Content Table
CREATE TABLE IF NOT EXISTS multimodal_content (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    content_modality VARCHAR(50) NOT NULL,
    original_format VARCHAR(50),
    original_size_bytes INTEGER,
    original_data BYTEA,
    processed_text TEXT,
    processed_format VARCHAR(50),
    processed_data JSONB,
    source_document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    page_number INTEGER,
    bounding_box JSONB,
    position_index INTEGER,
    extraction_method VARCHAR(100),
    extraction_confidence FLOAT,
    extraction_metadata JSONB DEFAULT '{}',
    context_before TEXT,
    context_after TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_multimodal_knowledge_item ON multimodal_content(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_modality ON multimodal_content(content_modality);
CREATE INDEX IF NOT EXISTS idx_multimodal_source_doc ON multimodal_content(source_document_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_extraction ON multimodal_content(extraction_method);

-- Knowledge Relationships Table
CREATE TABLE IF NOT EXISTS knowledge_relationships (
    id SERIAL PRIMARY KEY,
    from_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    to_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    bidirectional BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    UNIQUE(from_item_id, to_item_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_from ON knowledge_relationships(from_item_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_to ON knowledge_relationships(to_item_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_type ON knowledge_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_strength ON knowledge_relationships(strength DESC);

-- Table-Specific Storage
CREATE TABLE IF NOT EXISTS kb_tables (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    headers JSONB NOT NULL,
    data_types JSONB,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    markdown_representation TEXT,
    csv_data TEXT,
    json_data JSONB,
    has_header_row BOOLEAN DEFAULT true,
    is_numeric BOOLEAN DEFAULT false,
    has_totals BOOLEAN DEFAULT false,
    caption TEXT,
    footnotes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_tables_knowledge_item ON kb_tables(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_kb_tables_size ON kb_tables(row_count, column_count);

-- Image-Specific Storage
CREATE TABLE IF NOT EXISTS kb_images (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    width INTEGER,
    height INTEGER,
    format VARCHAR(50),
    file_size_bytes INTEGER,
    description TEXT,
    caption TEXT,
    alt_text TEXT,
    detected_objects JSONB,
    detected_text TEXT,
    image_data BYTEA,
    thumbnail_data BYTEA,
    storage_path VARCHAR(500),
    visual_embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_images_knowledge_item ON kb_images(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_kb_images_format ON kb_images(format);
CREATE INDEX IF NOT EXISTS idx_kb_images_visual_embedding ON kb_images USING ivfflat (visual_embedding vector_cosine_ops) WITH (lists = 50);

-- Formula-Specific Storage
CREATE TABLE IF NOT EXISTS kb_formulas (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE UNIQUE,
    latex TEXT NOT NULL,
    mathml TEXT,
    ascii_math TEXT,
    variables JSONB,
    operators JSONB,
    complexity_level VARCHAR(50),
    formula_type VARCHAR(100),
    domain VARCHAR(100),
    rendered_svg TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_formulas_knowledge_item ON kb_formulas(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_kb_formulas_type ON kb_formulas(formula_type);
CREATE INDEX IF NOT EXISTS idx_kb_formulas_domain ON kb_formulas(domain);

-- Knowledge Usage Analytics
CREATE TABLE IF NOT EXISTS knowledge_usage (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    context_type VARCHAR(100),
    query_text TEXT,
    relevance_score FLOAT,
    user_rating INTEGER,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_usage_item ON knowledge_usage(knowledge_item_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_usage_event ON knowledge_usage(event_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_usage_timestamp ON knowledge_usage(timestamp DESC);

-- Knowledge Collections
CREATE TABLE IF NOT EXISTS knowledge_collections (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    kb_type_id INTEGER REFERENCES kb_types(id),
    icon VARCHAR(50),
    color VARCHAR(50),
    visibility VARCHAR(50) DEFAULT 'private',
    owner_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_collection_items (
    collection_id INTEGER REFERENCES knowledge_collections(id) ON DELETE CASCADE,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    position INTEGER,
    added_at TIMESTAMP DEFAULT NOW(),
    added_by VARCHAR(255),
    PRIMARY KEY (collection_id, knowledge_item_id)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_collections_owner ON knowledge_collections(owner_id);
CREATE INDEX IF NOT EXISTS idx_collection_items_collection ON knowledge_collection_items(collection_id);
CREATE INDEX IF NOT EXISTS idx_collection_items_item ON knowledge_collection_items(knowledge_item_id);

-- ================================================================
-- MEMORY & KNOWLEDGE GRAPH TABLES
-- ================================================================

-- Memory Items table
CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id INTEGER REFERENCES agents(id),
    content VARCHAR NOT NULL,
    memory_type VARCHAR(100) NOT NULL,
    memory_level VARCHAR(50) DEFAULT 'working',
    importance FLOAT DEFAULT 0.5,
    embedding vector(1024),
    access_count INTEGER DEFAULT 0,
    last_access TIMESTAMP DEFAULT NOW(),
    decay_rate FLOAT DEFAULT 0.1,
    associations VARCHAR[],
    metadata JSON DEFAULT '{}'::json,
    created_at TIMESTAMP DEFAULT NOW(),
    success_rate FLOAT DEFAULT 0.0,
    usage_in_solutions INTEGER DEFAULT 0,
    average_retrieval_time FLOAT DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_memory_items_agent ON memory_items(agent_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_memory_level ON memory_items(memory_level);
CREATE INDEX IF NOT EXISTS idx_memory_items_embedding ON memory_items USING ivfflat (embedding vector_cosine_ops);

-- Knowledge Nodes table
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id),
    concept VARCHAR(255),
    description TEXT,
    node_type VARCHAR(50),
    embedding vector(1024),
    importance FLOAT,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_agent_id ON knowledge_nodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding ON knowledge_nodes USING ivfflat (embedding vector_cosine_ops);

-- Knowledge Edges table
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_node_id UUID REFERENCES knowledge_nodes(id),
    to_node_id UUID REFERENCES knowledge_nodes(id),
    relationship_type VARCHAR(50),
    strength FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- KNOWLEDGE GRAPH ENTITIES (Migration 007)
-- ================================================================

-- KB Entities Table (extracted entities from documents)
CREATE TABLE IF NOT EXISTS kb_entities (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    canonical_name VARCHAR(255),
    description TEXT,
    embedding vector(1024),
    mention_count INTEGER DEFAULT 0,
    importance_score FLOAT DEFAULT 0.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_kb_entities_canonical ON kb_entities(LOWER(canonical_name));
CREATE INDEX IF NOT EXISTS idx_kb_entities_type ON kb_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_kb_entities_importance ON kb_entities(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_kb_entities_embedding ON kb_entities USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_kb_entities_name_fts ON kb_entities USING gin(to_tsvector('english', entity_name || ' ' || COALESCE(description, '')));

-- Knowledge Entity Mentions (links entities to knowledge items)
CREATE TABLE IF NOT EXISTS knowledge_entity_mentions (
    id SERIAL PRIMARY KEY,
    knowledge_item_id INTEGER REFERENCES knowledge_items(id) ON DELETE CASCADE,
    entity_id INTEGER REFERENCES kb_entities(id) ON DELETE CASCADE,
    mention_context TEXT,
    confidence FLOAT DEFAULT 1.0,
    position_in_source INTEGER,
    extraction_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON knowledge_entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_knowledge ON knowledge_entity_mentions(knowledge_item_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_mentions_unique ON knowledge_entity_mentions(knowledge_item_id, entity_id, position_in_source);

-- Entity Relationships (entity-to-entity graph)
CREATE TABLE IF NOT EXISTS entity_relationships (
    id SERIAL PRIMARY KEY,
    from_entity_id INTEGER REFERENCES kb_entities(id) ON DELETE CASCADE,
    to_entity_id INTEGER REFERENCES kb_entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    evidence_source_id INTEGER REFERENCES knowledge_items(id) ON DELETE SET NULL,
    evidence_text TEXT,
    bidirectional BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_relationships_from ON entity_relationships(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_to ON entity_relationships(to_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_type ON entity_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_strength ON entity_relationships(strength DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_relationships_unique ON entity_relationships(from_entity_id, to_entity_id, relationship_type);

-- Entity Clusters (semantic clustering)
CREATE TABLE IF NOT EXISTS entity_clusters (
    id SERIAL PRIMARY KEY,
    cluster_name VARCHAR(255),
    cluster_topic VARCHAR(500),
    entity_ids INTEGER[],
    size INTEGER DEFAULT 0,
    coherence_score FLOAT DEFAULT 0.0,
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entity_clusters_size ON entity_clusters(size DESC);

-- External Knowledge table
CREATE TABLE IF NOT EXISTS external_knowledge (
    id SERIAL PRIMARY KEY,
    content JSON NOT NULL,
    source VARCHAR(255) NOT NULL DEFAULT 'external',
    knowledge_metadata JSON,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- MCP TOOLS & CREDENTIALS TABLES
-- ================================================================

-- MCP Tools table
CREATE TABLE IF NOT EXISTS mcp_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    mcp_server_url VARCHAR(500),
    capabilities JSON DEFAULT '{}',
    credentials_schema JSON DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    provider VARCHAR(255),
    version VARCHAR(50),
    icon VARCHAR(100),
    category VARCHAR(100),
    tags TEXT[],
    metadata JSON DEFAULT '{}',
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mcp_tools_status ON mcp_tools(status);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_category ON mcp_tools(category);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_provider ON mcp_tools(provider);

-- System Settings table (database-backed configuration management)
CREATE TABLE IF NOT EXISTS system_settings (
    id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    key VARCHAR(100) NOT NULL,
    value TEXT,
    value_type VARCHAR(20) DEFAULT 'string',
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    is_required BOOLEAN DEFAULT FALSE,
    default_value TEXT,
    validation_rules JSON,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system',
    UNIQUE(category, key)
);

CREATE INDEX IF NOT EXISTS idx_system_settings_category ON system_settings(category);
CREATE INDEX IF NOT EXISTS idx_system_settings_key ON system_settings(key);
CREATE INDEX IF NOT EXISTS idx_system_settings_category_key ON system_settings(category, key);

-- Credential Types table (definitions for credential schemas)
-- MUST be before credentials and agent_tool_assignments
CREATE TABLE IF NOT EXISTS credential_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    icon VARCHAR(50),
    description TEXT,
    schema_definition JSON NOT NULL,
    test_endpoint JSON,
    documentation_url VARCHAR(500),
    is_system BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_credential_types_id ON credential_types(id);
CREATE INDEX IF NOT EXISTS ix_credential_types_name ON credential_types(name);
CREATE INDEX IF NOT EXISTS ix_credential_types_category ON credential_types(category);
CREATE INDEX IF NOT EXISTS ix_credential_types_is_active ON credential_types(is_active);

-- Credentials table (stores encrypted credentials)
CREATE TABLE IF NOT EXISTS credentials (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    credential_type_id INTEGER REFERENCES credential_types(id) ON DELETE CASCADE NOT NULL,
    encrypted_data TEXT NOT NULL,
    environment VARCHAR(50) DEFAULT 'production',
    description TEXT,
    tags JSON DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    last_tested TIMESTAMP,
    test_status VARCHAR(50),
    test_message TEXT,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, environment)
);

CREATE INDEX idx_credentials_type ON credentials(credential_type_id);
CREATE INDEX idx_credentials_env ON credentials(environment);
CREATE INDEX idx_credentials_active ON credentials(is_active);

-- Agent Tool Assignments table
-- References credentials, so must come after
CREATE TABLE IF NOT EXISTS agent_tool_assignments (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE NOT NULL,
    tool_id INTEGER REFERENCES mcp_tools(id) ON DELETE CASCADE NOT NULL,
    credential_id INTEGER REFERENCES credentials(id) ON DELETE SET NULL,
    enabled BOOLEAN DEFAULT TRUE,
    permissions JSON DEFAULT '{}',
    configuration JSON DEFAULT '{}',
    assigned_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(agent_id, tool_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_tools_agent ON agent_tool_assignments(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tools_tool ON agent_tool_assignments(tool_id);
CREATE INDEX IF NOT EXISTS idx_agent_tools_enabled ON agent_tool_assignments(enabled);
CREATE INDEX IF NOT EXISTS idx_agent_tool_credential ON agent_tool_assignments(credential_id);

-- Tool Credentials table
CREATE TABLE IF NOT EXISTS tool_credentials (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER REFERENCES mcp_tools(id) ON DELETE CASCADE NOT NULL,
    agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    environment VARCHAR(50) NOT NULL,
    credentials JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tool_id, agent_id, environment)
);

CREATE INDEX IF NOT EXISTS idx_tool_credentials_tool ON tool_credentials(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_agent ON tool_credentials(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_env ON tool_credentials(environment);

-- Tool Usage Logs table
CREATE TABLE IF NOT EXISTS tool_usage_logs (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES workflow_executions(id),
    agent_id INTEGER REFERENCES agents(id) NOT NULL,
    tool_id INTEGER REFERENCES mcp_tools(id) NOT NULL,
    method_called VARCHAR(255),
    input_data JSONB,
    output_data JSONB,
    success BOOLEAN,
    execution_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_usage_agent ON tool_usage_logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_tool ON tool_usage_logs(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_execution ON tool_usage_logs(execution_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_success ON tool_usage_logs(success);

-- Tools table (UI registry)
CREATE TABLE IF NOT EXISTS tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    description TEXT,
    category VARCHAR,
    provider VARCHAR,
    version VARCHAR,
    icon VARCHAR,
    pricing VARCHAR,
    tags JSON DEFAULT '[]',
    permissions JSON DEFAULT '[]',
    required_credentials JSON DEFAULT '[]',
    supported_environments JSON DEFAULT '[]',
    mcp_config JSON,
    status VARCHAR DEFAULT 'available',
    is_installed BOOLEAN DEFAULT FALSE,
    installation_date TIMESTAMP,
    last_used TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    rating FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP
);

-- Tool Configurations table
CREATE TABLE IF NOT EXISTS tool_configurations (
    id SERIAL PRIMARY KEY,
    tool_id VARCHAR NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agent Tool Permissions table
CREATE TABLE IF NOT EXISTS agent_tool_permissions (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    tool_id VARCHAR(255) NOT NULL,
    environment VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Credential Audit Logs table
CREATE TABLE IF NOT EXISTS credential_audit_logs (
    id SERIAL PRIMARY KEY,
    credential_id INTEGER REFERENCES credentials(id) ON DELETE CASCADE,
    tool_id INTEGER,
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW() NOT NULL,
    details JSON,
    metadata JSON,
    ip_address VARCHAR(45),
    user_agent VARCHAR,
    success BOOLEAN,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_credential_audit_credential ON credential_audit_logs(credential_id);
CREATE INDEX IF NOT EXISTS idx_credential_audit_action ON credential_audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_credential_audit_created ON credential_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_credential_audit_user ON credential_audit_logs(user_id);

-- Permission Audit Logs table
CREATE TABLE IF NOT EXISTS permission_audit_logs (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    details JSON
);

-- ================================================================
-- LLM MODELS REGISTRY (PRD-15)
-- ================================================================

CREATE TABLE IF NOT EXISTS llm_models (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    model_family VARCHAR(100),
    capabilities JSONB DEFAULT '{}'::jsonb,
    context_window INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    supports_functions BOOLEAN DEFAULT FALSE,
    supports_vision BOOLEAN DEFAULT FALSE,
    supports_streaming BOOLEAN DEFAULT TRUE,
    input_cost_per_1k_tokens FLOAT,
    output_cost_per_1k_tokens FLOAT,
    description TEXT,
    release_date TIMESTAMP,
    deprecation_date TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    recommended_for JSONB DEFAULT '[]'::jsonb,
    default_temperature FLOAT DEFAULT 0.7,
    min_temperature FLOAT DEFAULT 0.0,
    max_temperature FLOAT DEFAULT 2.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_models_provider ON llm_models(provider);
CREATE INDEX IF NOT EXISTS idx_llm_models_status ON llm_models(status);

-- ================================================================
-- CONTEXT ENGINEERING TABLES (PRD-03)
-- ================================================================

-- Context Policies table
CREATE TABLE IF NOT EXISTS context_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id VARCHAR(128) NOT NULL,
    domain VARCHAR(128),
    agent_id VARCHAR(128),
    tenant_id VARCHAR(128),
    slots JSONB NOT NULL,
    max_total_chars INTEGER NOT NULL DEFAULT 12000,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_context_policies_policy_id ON context_policies(policy_id);
CREATE INDEX IF NOT EXISTS idx_context_policies_domain ON context_policies(domain);

-- Context Templates table
CREATE TABLE IF NOT EXISTS context_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    template_type VARCHAR(100),
    template_content TEXT,
    variables JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Context Examples table
CREATE TABLE IF NOT EXISTS context_examples (
    id SERIAL PRIMARY KEY,
    example_type VARCHAR(100),
    input_text TEXT,
    output_text TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context Queries table
CREATE TABLE IF NOT EXISTS context_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_type VARCHAR(50),
    results_count INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context Sources table
CREATE TABLE IF NOT EXISTS context_sources (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(50),
    source_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context Usage table
CREATE TABLE IF NOT EXISTS context_usage (
    id SERIAL PRIMARY KEY,
    document_id INTEGER,
    chunk_id INTEGER,
    query_text TEXT,
    relevance_score FLOAT,
    used_in_response BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Context Permissions table
CREATE TABLE IF NOT EXISTS context_permissions (
    id SERIAL PRIMARY KEY,
    context_id UUID,
    agent_id INTEGER REFERENCES agents(id),
    permission_level VARCHAR(50) DEFAULT 'read',
    granted_at TIMESTAMP DEFAULT NOW(),
    granted_by INTEGER REFERENCES agents(id)
);

-- Context Optimizations table
CREATE TABLE IF NOT EXISTS context_optimizations (
    id SERIAL PRIMARY KEY,
    optimization_type VARCHAR(100),
    parameters JSON,
    results JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context Patterns table
CREATE TABLE IF NOT EXISTS context_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(255),
    pattern_data JSONB,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- CODEGRAPH TABLES (PRD-11)
-- ================================================================

-- Code Symbols table
CREATE TABLE IF NOT EXISTS code_symbols (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project VARCHAR(128) NOT NULL,
    file_path TEXT NOT NULL,
    symbol_name VARCHAR(256) NOT NULL,
    symbol_type VARCHAR(32) NOT NULL,
    signature TEXT,
    docstring TEXT,
    start_line INTEGER,
    end_line INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_code_symbols_project_name ON code_symbols(project, symbol_name);

-- Code Edges table
CREATE TABLE IF NOT EXISTS code_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project VARCHAR(128) NOT NULL,
    src_symbol_id UUID REFERENCES code_symbols(id) ON DELETE CASCADE NOT NULL,
    dst_symbol_id UUID REFERENCES code_symbols(id) ON DELETE CASCADE NOT NULL,
    edge_type VARCHAR(32) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_code_edges_project_type ON code_edges(project, edge_type);

-- ================================================================
-- PLAYBOOKS TABLE (PRD-08)
-- ================================================================

CREATE TABLE IF NOT EXISTS playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(128) NOT NULL,
    tenant_id VARCHAR(128),
    pattern JSONB NOT NULL,
    support INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_playbooks_name ON playbooks(name);
CREATE INDEX IF NOT EXISTS idx_playbooks_tenant ON playbooks(tenant_id);

-- ================================================================
-- INTER-AGENT COMMUNICATION TABLES (PRD-04)
-- ================================================================

-- Agent Messages table
CREATE TABLE IF NOT EXISTS agent_messages (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::varchar,
    from_agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    to_agent_id INTEGER REFERENCES agents(id) ON DELETE CASCADE,
    message_type VARCHAR(50) NOT NULL,
    content JSON NOT NULL,
    priority INTEGER DEFAULT 5,
    status VARCHAR(50) DEFAULT 'sent',
    requires_ack BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    delivered_at TIMESTAMP,
    read_at TIMESTAMP,
    acknowledged_at TIMESTAMP
);

-- Shared Contexts table
CREATE TABLE IF NOT EXISTS shared_contexts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255),
    team_id VARCHAR(255),
    context_data JSON NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by INTEGER REFERENCES agents(id)
);

-- Collaboration Sessions table
CREATE TABLE IF NOT EXISTS collaboration_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_id INTEGER,
    problem_description TEXT,
    team_agents JSON,
    strategy VARCHAR(50) DEFAULT 'ensemble',
    shared_context_id UUID,
    status VARCHAR(50) DEFAULT 'pending',
    result JSON,
    consensus_data JSON,
    metrics JSON,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Collaboration Proposals table
CREATE TABLE IF NOT EXISTS collaboration_proposals (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES collaboration_sessions(id) ON DELETE CASCADE,
    agent_id INTEGER REFERENCES agents(id),
    proposal_type VARCHAR(50),
    proposal_content JSON NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    tokens_used INTEGER,
    execution_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Consensus Votes table
CREATE TABLE IF NOT EXISTS consensus_votes (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES collaboration_sessions(id) ON DELETE CASCADE,
    proposal_id INTEGER REFERENCES collaboration_proposals(id) ON DELETE CASCADE,
    agent_id INTEGER REFERENCES agents(id),
    vote_weight FLOAT DEFAULT 1.0,
    vote_value VARCHAR(50),
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Message Broadcasts table
CREATE TABLE IF NOT EXISTS message_broadcasts (
    id VARCHAR(255) PRIMARY KEY DEFAULT uuid_generate_v4()::varchar,
    from_agent_id INTEGER REFERENCES agents(id),
    team_agents JSON,
    message_type VARCHAR(50),
    content JSON,
    priority INTEGER DEFAULT 5,
    delivered_to JSON,
    failed_deliveries JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- TASK DECOMPOSITION & COORDINATION TABLES
-- ================================================================

-- Tasks table
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL,
    owner_id INTEGER NOT NULL,
    immediate_memory JSON,
    working_memory JSON,
    short_term_memory JSON,
    long_term_memory JSON,
    importance FLOAT,
    tools JSON,
    tool_scores JSON,
    dependencies JSON,
    execution_status JSON,
    reasoning JSON,
    augmented_memory JSON,
    similarity_score FLOAT,
    consensus_score FLOAT,
    coordination JSON,
    optimization JSON,
    optimization_config JSON,
    field_value FLOAT,
    influence_weights JSON,
    gradient JSON,
    field_timestamp TIMESTAMP,
    propagation_timestamp TIMESTAMP,
    interactions JSON,
    emergent_effect FLOAT,
    embeddings JSON,
    stability FLOAT,
    prev_field_value FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Task Decompositions table
CREATE TABLE IF NOT EXISTS task_decompositions (
    id SERIAL PRIMARY KEY,
    task_id INTEGER,
    decomposition_data JSONB,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Task Assignments table
CREATE TABLE IF NOT EXISTS task_assignments (
    id SERIAL PRIMARY KEY,
    task_id INTEGER,
    agent_id INTEGER REFERENCES agents(id),
    assigned_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(50)
);

-- Agent Coordination table
CREATE TABLE IF NOT EXISTS agent_coordination (
    id SERIAL PRIMARY KEY,
    coordination_type VARCHAR(100),
    participating_agents JSON,
    coordination_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent Runtimes table
CREATE TABLE IF NOT EXISTS agent_runtimes (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    runtime_config JSON,
    status VARCHAR(50),
    started_at TIMESTAMP,
    stopped_at TIMESTAMP
);

-- Execution Contexts table
CREATE TABLE IF NOT EXISTS execution_contexts (
    id SERIAL PRIMARY KEY,
    context_name VARCHAR(255),
    context_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- ANALYTICS & MONITORING TABLES (PRD-06)
-- ================================================================

-- Dashboard Configs table
CREATE TABLE IF NOT EXISTS dashboard_configs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    config_name VARCHAR(255),
    config_data JSON,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Analytics Snapshots table
CREATE TABLE IF NOT EXISTS analytics_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_type VARCHAR(100),
    snapshot_data JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

-- System Metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(100),
    metric_name VARCHAR(255),
    metric_value FLOAT,
    metric_unit VARCHAR(50),
    metric_data JSON,
    timestamp TIMESTAMP DEFAULT NOW(),
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Context Optimization Metrics table
CREATE TABLE IF NOT EXISTS context_optimization_metrics (
    id SERIAL PRIMARY KEY,
    tokens_saved INTEGER DEFAULT 0,
    compression_ratio FLOAT DEFAULT 1.0,
    optimization_type VARCHAR(100),
    context_size_before INTEGER,
    context_size_after INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_context_optimization_created_at ON context_optimization_metrics(created_at);

-- Custom Metrics table
CREATE TABLE IF NOT EXISTS custom_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255),
    metric_value FLOAT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Alert Configs table
CREATE TABLE IF NOT EXISTS alert_configs (
    id SERIAL PRIMARY KEY,
    alert_name VARCHAR(255),
    alert_condition JSONB,
    alert_actions JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- LEARNING & PERFORMANCE TABLES (PRD-05)
-- ================================================================

-- Agent Performance table
CREATE TABLE IF NOT EXISTS agent_performance (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    performance_data JSONB,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Agent Performance Tracking table
CREATE TABLE IF NOT EXISTS agent_performance_tracking (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Agent Behavior Monitoring table
CREATE TABLE IF NOT EXISTS agent_behavior_monitoring (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    behavior_data JSONB,
    anomaly_detected BOOLEAN DEFAULT FALSE,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Learning Outcomes table
CREATE TABLE IF NOT EXISTS learning_outcomes (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    outcome_type VARCHAR(100),
    outcome_data JSONB,
    success_rate FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Learning Progress Tracking table
CREATE TABLE IF NOT EXISTS learning_progress_tracking (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    knowledge_items INTEGER DEFAULT 0,
    memory_consolidations INTEGER DEFAULT 0,
    performance_improvement FLOAT DEFAULT 0.0,
    knowledge_transfers INTEGER DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_learning_progress_agent_time ON learning_progress_tracking(agent_id, recorded_at);

-- Multi-Agent Reasoning table
CREATE TABLE IF NOT EXISTS multi_agent_reasoning (
    id SERIAL PRIMARY KEY,
    reasoning_type VARCHAR(100),
    participating_agents JSON,
    reasoning_data JSONB,
    consensus_reached BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Field Interactions table (Field Theory)
CREATE TABLE IF NOT EXISTS field_interactions (
    id SERIAL PRIMARY KEY,
    interaction_type VARCHAR(100),
    field_data JSONB,
    strength FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Field States table
CREATE TABLE IF NOT EXISTS field_states (
    id SERIAL PRIMARY KEY,
    state_name VARCHAR(255),
    state_data JSONB,
    stability FLOAT,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- EVALUATION & ASSESSMENT TABLES
-- ================================================================

-- Evaluation Results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    evaluation_id VARCHAR(255) UNIQUE NOT NULL,
    evaluation_type VARCHAR(100) NOT NULL,
    scope VARCHAR(100) NOT NULL,
    target_id VARCHAR(255) NOT NULL,
    overall_score FLOAT NOT NULL,
    detailed_results JSON,
    success BOOLEAN,
    error_message TEXT,
    execution_time_seconds FLOAT,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Benchmark Assessments table
CREATE TABLE IF NOT EXISTS benchmark_assessments (
    id SERIAL PRIMARY KEY,
    benchmark_id VARCHAR(255) NOT NULL,
    benchmark_name VARCHAR(255) NOT NULL,
    benchmark_type VARCHAR(100) NOT NULL,
    validity_score FLOAT,
    reliability_score FLOAT,
    discriminatory_power FLOAT,
    overall_quality FLOAT,
    quality_classification VARCHAR(50),
    assessment_data JSON,
    recommendations JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Component Metrics table
CREATE TABLE IF NOT EXISTS component_metrics (
    id SERIAL PRIMARY KEY,
    component_id VARCHAR(255) NOT NULL,
    component_type VARCHAR(100) NOT NULL,
    performance_score FLOAT,
    reliability_score FLOAT,
    readiness_score FLOAT,
    capability_rating FLOAT,
    complexity_index FLOAT,
    environment_factor FLOAT,
    assessment_details JSON,
    assessment_timestamp TIMESTAMP DEFAULT NOW()
);

-- Integration Analysis table
CREATE TABLE IF NOT EXISTS integration_analysis (
    id SERIAL PRIMARY KEY,
    system_id VARCHAR(255) NOT NULL,
    coherence_score FLOAT,
    efficiency_score FLOAT,
    emergence_score FLOAT,
    integration_score FLOAT,
    integration_classification VARCHAR(50),
    analysis_data JSON,
    recommendations JSON,
    confidence_level FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- CONFIGURATION TABLES
-- ================================================================

-- System Configurations table
CREATE TABLE IF NOT EXISTS system_configurations (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSON,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    updated_by VARCHAR(255)
);

-- RAG Configurations table
CREATE TABLE IF NOT EXISTS rag_configurations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding_model VARCHAR(255),
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    retrieval_strategy VARCHAR(100) DEFAULT 'similarity',
    top_k INTEGER DEFAULT 5,
    similarity_threshold FLOAT DEFAULT 0.7,
    configuration JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Schema Versions table (for tracking migrations)
CREATE TABLE IF NOT EXISTS schema_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    applied_at TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- DATABASE KNOWLEDGE SOURCE TABLES (PRD-21)
-- ================================================================

-- Main database knowledge source table
CREATE TABLE IF NOT EXISTS database_knowledge_sources (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    -- Uses existing credential system by name
    credential_name VARCHAR(255),
    credential_environment VARCHAR(50) DEFAULT 'production',
    dialect VARCHAR(50) NOT NULL CHECK (dialect IN ('postgresql', 'mysql', 'sqlite', 'mssql', 'snowflake', 'bigquery', 'redshift')),
    connection_pool_size INTEGER DEFAULT 5,
    max_rows_limit INTEGER DEFAULT 10000,
    query_timeout_seconds INTEGER DEFAULT 30,
    schema_metadata JSONB,
    schema_version INTEGER DEFAULT 1,
    schema_hash VARCHAR(64),
    last_introspected TIMESTAMP,
    semantic_layer JSONB,
    schema_cache_ttl INTEGER DEFAULT 3600,
    query_cache_ttl INTEGER DEFAULT 300,
    total_queries_executed INTEGER DEFAULT 0,
    avg_query_time_ms FLOAT,
    last_successful_query TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'active',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, name)
);

CREATE INDEX idx_dks_tenant_name ON database_knowledge_sources(tenant_id, name);
CREATE INDEX idx_dks_credential_lookup ON database_knowledge_sources(credential_name, credential_environment);
CREATE INDEX idx_dks_status ON database_knowledge_sources(status, is_active);

-- Table relationships for JOIN optimization
CREATE TABLE IF NOT EXISTS database_relationships (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    from_table VARCHAR(255) NOT NULL,
    from_column VARCHAR(255) NOT NULL,
    to_table VARCHAR(255) NOT NULL,
    to_column VARCHAR(255) NOT NULL,
    relationship_type VARCHAR(50),
    is_inferred BOOLEAN DEFAULT FALSE,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dr_source ON database_relationships(source_id);
CREATE INDEX idx_dr_tables ON database_relationships(from_table, to_table);

-- Query audit trail
CREATE TABLE IF NOT EXISTS database_query_audit (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    user_id INTEGER,
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    natural_language_query TEXT NOT NULL,
    generated_sql TEXT,
    validated_sql TEXT,
    execution_time_ms INTEGER,
    row_count INTEGER,
    bytes_processed INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    validation_errors JSONB,
    was_cached BOOLEAN DEFAULT FALSE,
    cache_key VARCHAR(64),
    visualization_type VARCHAR(50),
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dqa_tenant_source ON database_query_audit(tenant_id, source_id);
CREATE INDEX idx_dqa_user ON database_query_audit(user_id);
CREATE INDEX idx_dqa_created ON database_query_audit(created_at DESC);

-- Semantic metrics
CREATE TABLE IF NOT EXISTS semantic_metrics (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    sql_expression TEXT NOT NULL,
    aggregation VARCHAR(50),
    format VARCHAR(50),
    description TEXT,
    business_definition TEXT,
    tables_used JSONB,
    drill_down_dimensions JSONB,
    supports_time_grain BOOLEAN DEFAULT TRUE,
    default_time_grain VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    is_featured BOOLEAN DEFAULT FALSE,
    is_certified BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, name)
);

CREATE INDEX idx_sm_source ON semantic_metrics(source_id);
CREATE INDEX idx_sm_featured ON semantic_metrics(is_featured, is_certified);

-- Semantic dimensions
CREATE TABLE IF NOT EXISTS semantic_dimensions (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    tenant_id INTEGER NOT NULL,
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    sql_expression TEXT NOT NULL,
    type VARCHAR(50),
    description TEXT,
    hierarchy_levels JSONB,
    parent_dimension_id INTEGER REFERENCES semantic_dimensions(id),
    cached_values JSONB,
    total_unique_values INTEGER,
    is_featured BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, name)
);

CREATE INDEX idx_sd_source ON semantic_dimensions(source_id);
CREATE INDEX idx_sd_parent ON semantic_dimensions(parent_dimension_id);

-- Query templates
CREATE TABLE IF NOT EXISTS database_query_templates (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES database_knowledge_sources(id) ON DELETE CASCADE,
    dialect VARCHAR(50),
    category VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    natural_language TEXT NOT NULL,
    sql_template TEXT,
    parameters JSONB,
    visualization_type VARCHAR(50),
    usage_count INTEGER DEFAULT 0,
    avg_execution_time_ms FLOAT,
    tags JSONB,
    is_featured BOOLEAN DEFAULT FALSE,
    is_certified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dqt_source ON database_query_templates(source_id);
CREATE INDEX idx_dqt_dialect ON database_query_templates(dialect, category);
CREATE INDEX idx_dqt_featured ON database_query_templates(is_featured);

-- ================================================================
-- INSERT INITIAL DATA
-- ================================================================

-- Insert schema version
INSERT INTO schema_versions (version, description) 
VALUES ('1.0.0', 'Complete schema initialization - October 2025')
ON CONFLICT DO NOTHING;

-- Insert system agent for chatbot memory (required for memory_items foreign key)
INSERT INTO agents (id, name, agent_type, status, configuration, created_at, updated_at)
VALUES (1, 'ChatMemory', 'system', 'active', '{}', NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Insert default system configurations
INSERT INTO system_configurations (config_key, config_value, description) VALUES
('system.max_agents', '{"value": 100}', 'Maximum number of agents allowed'),
('system.default_timeout', '{"value": 300}', 'Default timeout for operations (seconds)'),
('rag.default_model', '{"value": "text-embedding-ada-002"}', 'Default embedding model'),
('workflow.max_concurrent', '{"value": 10}', 'Maximum concurrent workflow executions')
ON CONFLICT (config_key) DO NOTHING;

-- Insert default RAG configuration
INSERT INTO rag_configurations (name, embedding_model, created_by) VALUES
('default', 'text-embedding-ada-002', 'system')
ON CONFLICT DO NOTHING;

-- Insert sample LLM models
INSERT INTO llm_models (provider, model_id, display_name, model_family, context_window, max_output_tokens, 
                       input_cost_per_1k_tokens, output_cost_per_1k_tokens, status) VALUES
('openai', 'gpt-4-turbo-preview', 'GPT-4 Turbo', 'gpt-4', 128000, 4096, 0.01, 0.03, 'active'),
('openai', 'gpt-4', 'GPT-4', 'gpt-4', 8192, 4096, 0.03, 0.06, 'active'),
('openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 'gpt-3.5', 16385, 4096, 0.0005, 0.0015, 'active'),
('anthropic', 'claude-3-opus-20240229', 'Claude 3 Opus', 'claude-3', 200000, 4096, 0.015, 0.075, 'active'),
('anthropic', 'claude-3-sonnet-20240229', 'Claude 3 Sonnet', 'claude-3', 200000, 4096, 0.003, 0.015, 'active')
ON CONFLICT (model_id) DO NOTHING;

-- Insert sample database query templates
INSERT INTO database_query_templates (
    dialect, category, name, description, natural_language, sql_template, 
    parameters, visualization_type, tags, is_featured, is_certified
) VALUES 
(
    'postgresql', 'analytics', 'Top N by Revenue',
    'Show top customers by revenue with time filter',
    'Show top {n} customers by revenue in the last {days} days',
    'SELECT customer_id, customer_name, SUM(amount) as total_revenue FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL ''{days} days'' GROUP BY customer_id, customer_name ORDER BY total_revenue DESC LIMIT {n}',
    '[{"name": "n", "type": "integer", "default": 10}, {"name": "days", "type": "integer", "default": 30}]'::jsonb,
    'bar',
    '["revenue", "customers", "top-n"]'::jsonb,
    true, true
),
(
    'postgresql', 'analytics', 'Revenue Trend',
    'Revenue trend over time',
    'Show revenue trend over the last {months} months',
    'SELECT DATE_TRUNC(''month'', order_date) as month, SUM(amount) as revenue FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL ''{months} months'' GROUP BY month ORDER BY month',
    '[{"name": "months", "type": "integer", "default": 12}]'::jsonb,
    'line',
    '["revenue", "trend", "time-series"]'::jsonb,
    true, true
),
(
    'postgresql', 'reporting', 'Daily Summary',
    'Daily business metrics summary',
    'Daily business summary for {date}',
    'SELECT COUNT(DISTINCT order_id) as total_orders, COUNT(DISTINCT customer_id) as unique_customers, SUM(amount) as revenue, AVG(amount) as avg_order_value FROM orders WHERE DATE(order_date) = ''{date}''',
    '[{"name": "date", "type": "date", "default": "CURRENT_DATE"}]'::jsonb,
    'table',
    '["daily", "summary", "kpi"]'::jsonb,
    true, true
)
ON CONFLICT DO NOTHING;

-- ================================================================
-- COMMENTS FOR DOCUMENTATION
-- ================================================================

COMMENT ON DATABASE orchestrator_db IS 'Automatos AI Platform - Complete Database Schema';

COMMENT ON TABLE agents IS 'AI agents with configuration and performance tracking';
COMMENT ON TABLE workflows IS 'Workflow definitions and configurations';
COMMENT ON TABLE documents IS 'Document storage and tracking';
COMMENT ON TABLE document_chunks IS 'Chunked documents with vector embeddings for semantic search';
COMMENT ON TABLE memory_items IS 'Agent memory system with hierarchical storage';
COMMENT ON TABLE knowledge_nodes IS 'Knowledge graph nodes with semantic embeddings';
COMMENT ON TABLE mcp_tools IS 'MCP (Model Context Protocol) tools registry';
COMMENT ON TABLE context_policies IS 'Context engineering policies for optimal prompt construction';
COMMENT ON TABLE llm_models IS 'Registry of available LLM models from different providers';

-- ================================================================
-- SCHEMA INITIALIZATION COMPLETE
-- ================================================================
