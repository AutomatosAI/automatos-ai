-- PostgreSQL Database Initialization Script
-- This script sets up the initial database structure for Automatos AI

-- Create database if it doesn't exist (this is handled by POSTGRES_DB env var)
-- CREATE DATABASE IF NOT EXISTS automatos_ai;

-- Set timezone
SET timezone = 'UTC';

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create initial schema (tables will be created by Alembic migrations)
-- This file ensures the database is ready for migrations

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL database initialized successfully for Automatos AI';
END $$;-- Tool Tables Creation Script
-- Creates all missing Tool-related tables for Automatos AI

-- Tools table
CREATE TABLE IF NOT EXISTS tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    provider VARCHAR(255) NOT NULL,
    version VARCHAR(100) NOT NULL,
    icon VARCHAR(50),
    pricing VARCHAR(100),
    rating REAL DEFAULT 0.0,
    tags TEXT[],
    status VARCHAR(50) DEFAULT 'available',
    is_installed BOOLEAN DEFAULT FALSE,
    is_configured BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    permissions TEXT[],
    required_credentials TEXT[],
    supported_environments TEXT[],
    mcp_config JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Tool Credentials table
CREATE TABLE IF NOT EXISTS tool_credentials (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    credential_key VARCHAR(100) NOT NULL,
    credential_value TEXT NOT NULL,
    environment VARCHAR(50) NOT NULL,
    description TEXT,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tool Configurations table
CREATE TABLE IF NOT EXISTS tool_configurations (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    environment VARCHAR(50) NOT NULL,
    configuration JSON DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    last_health_check TIMESTAMP,
    health_status VARCHAR(50) DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Agent Tool Permissions table
CREATE TABLE IF NOT EXISTS agent_tool_permissions (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    environment VARCHAR(50) NOT NULL,
    permissions TEXT[],
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tool Usage Logs table
CREATE TABLE IF NOT EXISTS tool_usage_logs (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    action VARCHAR(100) NOT NULL,
    agent_id INTEGER REFERENCES agents(id),
    environment VARCHAR(50),
    response_time_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Credential Audit Logs table
CREATE TABLE IF NOT EXISTS credential_audit_logs (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    credential_key VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    environment VARCHAR(50),
    user_id VARCHAR(255),
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Permission Audit Logs table
CREATE TABLE IF NOT EXISTS permission_audit_logs (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id),
    tool_id INTEGER NOT NULL REFERENCES tools(id),
    action VARCHAR(100) NOT NULL,
    environment VARCHAR(50),
    user_id VARCHAR(255),
    details JSON DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);
CREATE INDEX IF NOT EXISTS idx_tools_status ON tools(status);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_tool_id ON tool_credentials(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_credentials_environment ON tool_credentials(environment);
CREATE INDEX IF NOT EXISTS idx_tool_configurations_tool_id ON tool_configurations(tool_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_permissions_agent_id ON agent_tool_permissions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tool_permissions_tool_id ON agent_tool_permissions(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_logs_tool_id ON tool_usage_logs(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_logs_timestamp ON tool_usage_logs(timestamp);

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Tool tables created successfully';
END $$;

-- =====================================
-- API TESTING FIXES - Added Thu Sep 18 20:44:53 WEST 2025
-- =====================================

-- Fix missing original_filename column (causing 500 errors)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_filename VARCHAR(255);

-- Fix missing document columns
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_path VARCHAR(500);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);

-- Fix JSON syntax for system configurations
INSERT INTO system_configurations (config_key, config_value, description) VALUES
('system.max_agents', '{"value": 100}', 'Maximum number of agents allowed in the system'),
('system.default_timeout', '{"value": 300}', 'Default timeout for agent operations (seconds)'),
('rag.default_model', '{"value": "sentence-transformers/all-MiniLM-L6-v2"}', 'Default embedding model for RAG system'),
('workflow.max_concurrent', '{"value": 10}', 'Maximum concurrent workflow executions')
ON CONFLICT (config_key) DO UPDATE SET 
    config_value = EXCLUDED.config_value,
    updated_at = NOW();

-- Fix RAG configuration
INSERT INTO rag_configurations (name, embedding_model, chunk_size, chunk_overlap, retrieval_strategy, top_k, similarity_threshold, is_active, created_by)
VALUES ('default', 'sentence-transformers/all-MiniLM-L6-v2', 1000, 200, 'similarity', 5, 0.7, TRUE, 'system')
ON CONFLICT (name) DO UPDATE SET 
    embedding_model = EXCLUDED.embedding_model,
    updated_at = NOW();

-- Log API testing fixes
DO $$
BEGIN
    RAISE NOTICE 'API Testing database fixes applied successfully!';
    RAISE NOTICE 'Fixed 422 validation errors and 500 database errors';
END $$;
