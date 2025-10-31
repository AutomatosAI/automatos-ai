-- Migration: Add Workflow Templates Table
-- Date: 2025-10-01
-- Purpose: Store and manage workflow templates with full CRUD operations

-- Create workflow_templates table
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
    is_public BOOLEAN DEFAULT true,
    is_featured BOOLEAN DEFAULT false,
    is_system BOOLEAN DEFAULT false,
    icon VARCHAR(50),
    preview_image VARCHAR(500),
    documentation_url VARCHAR(500),
    version VARCHAR(50) DEFAULT '1.0',
    changelog JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    last_used_at TIMESTAMP,
    CONSTRAINT unique_template_id UNIQUE (template_id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_workflow_templates_category ON workflow_templates(category);
CREATE INDEX IF NOT EXISTS idx_workflow_templates_template_id ON workflow_templates(template_id);
CREATE INDEX IF NOT EXISTS idx_workflow_templates_is_public ON workflow_templates(is_public);
CREATE INDEX IF NOT EXISTS idx_workflow_templates_is_featured ON workflow_templates(is_featured);
CREATE INDEX IF NOT EXISTS idx_workflow_templates_popularity ON workflow_templates(popularity DESC);
CREATE INDEX IF NOT EXISTS idx_workflow_templates_created_at ON workflow_templates(created_at DESC);

-- Add comments for documentation
COMMENT ON TABLE workflow_templates IS 'Workflow templates that users can browse and use to create workflows';
COMMENT ON COLUMN workflow_templates.template_id IS 'Unique identifier for the template (e.g., ai-code-review)';
COMMENT ON COLUMN workflow_templates.template_definition IS 'JSON structure containing steps, agents, config, and variables';
COMMENT ON COLUMN workflow_templates.recommended_agents IS 'Array of agent types that work well with this template';
COMMENT ON COLUMN workflow_templates.use_count IS 'Number of workflows created from this template';
COMMENT ON COLUMN workflow_templates.success_rate IS 'Average success rate of workflows using this template';
COMMENT ON COLUMN workflow_templates.popularity IS 'Popularity score from 0-100 based on usage and ratings';
COMMENT ON COLUMN workflow_templates.is_system IS 'System templates cannot be deleted by users';
COMMENT ON COLUMN workflow_templates.is_featured IS 'Featured templates shown prominently in UI';

-- Insert default system templates
INSERT INTO workflow_templates (
    template_id, name, description, category, tags, difficulty,
    template_definition, recommended_agents, estimated_time,
    use_count, success_rate, popularity, is_public, is_featured, is_system,
    icon, created_by, version
) VALUES 
(
    'customer-support-automation',
    'Customer Support Automation',
    'Automated customer inquiry handling with AI-powered responses and escalation',
    'Support',
    '["customer-support", "automation", "ai-assistance"]'::jsonb,
    'beginner',
    '{
        "steps": [
            {"id": "classify", "name": "Classify Inquiry", "type": "classification"},
            {"id": "analyze", "name": "Analyze Context", "type": "analysis"},
            {"id": "generate", "name": "Generate Response", "type": "generation"},
            {"id": "review", "name": "Quality Review", "type": "review"},
            {"id": "send", "name": "Send Response", "type": "action"}
        ],
        "agents": ["customer-support", "quality-reviewer", "response-generator"],
        "config": {
            "auto_escalate": true,
            "response_tone": "friendly",
            "max_response_time": "5m"
        }
    }'::jsonb,
    '["customer-support", "quality-reviewer", "response-generator"]'::jsonb,
    '3-5 minutes',
    0, 0.0, 85, true, true, true,
    '💬', 'system', '1.0'
),
(
    'data-analysis-pipeline',
    'Data Analysis Pipeline',
    'Complete data analysis workflow from ingestion to visualization',
    'Data Processing',
    '["data-analysis", "visualization", "reporting"]'::jsonb,
    'intermediate',
    '{
        "steps": [
            {"id": "ingest", "name": "Data Ingestion", "type": "ingestion"},
            {"id": "clean", "name": "Data Cleaning", "type": "transformation"},
            {"id": "analyze", "name": "Statistical Analysis", "type": "analysis"},
            {"id": "visualize", "name": "Create Visualizations", "type": "visualization"}
        ],
        "agents": ["data-analyst", "data-engineer"],
        "config": {
            "data_sources": ["csv", "json", "database"],
            "output_formats": ["pdf", "html", "dashboard"]
        }
    }'::jsonb,
    '["data-analyst", "data-engineer"]'::jsonb,
    '10-15 minutes',
    0, 0.0, 90, true, true, true,
    '📊', 'system', '1.0'
),
(
    'content-generation',
    'AI Content Generation',
    'Generate high-quality content with AI assistance and human review',
    'Content',
    '["content", "ai-generation", "writing"]'::jsonb,
    'beginner',
    '{
        "steps": [
            {"id": "brief", "name": "Analyze Brief", "type": "analysis"},
            {"id": "research", "name": "Research Topics", "type": "research"},
            {"id": "generate", "name": "Generate Content", "type": "generation"}
        ],
        "agents": ["content-writer", "researcher"],
        "config": {
            "content_types": ["blog", "social", "email"],
            "tone": "professional",
            "word_count": 500
        }
    }'::jsonb,
    '["content-writer", "researcher"]'::jsonb,
    '5-8 minutes',
    0, 0.0, 75, true, true, true,
    '✍️', 'system', '1.0'
),
(
    'security-audit',
    'Security Audit',
    'Complete security assessment and compliance check',
    'Security',
    '["security", "audit", "compliance"]'::jsonb,
    'advanced',
    '{
        "steps": [
            {"id": "scan", "name": "Vulnerability Scan", "type": "scanning"},
            {"id": "review", "name": "Code Security Review", "type": "review"},
            {"id": "infrastructure", "name": "Infrastructure Audit", "type": "audit"},
            {"id": "compliance", "name": "Compliance Check", "type": "compliance"},
            {"id": "report", "name": "Report Generation", "type": "reporting"}
        ],
        "agents": ["security-guard"],
        "config": {
            "scan_depth": "comprehensive",
            "compliance_standards": ["OWASP", "GDPR", "SOC2"]
        }
    }'::jsonb,
    '["security-guard"]'::jsonb,
    '20-30 minutes',
    0, 0.0, 80, true, true, true,
    '🔒', 'system', '1.0'
);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_workflow_template_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER workflow_templates_updated_at
    BEFORE UPDATE ON workflow_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_workflow_template_updated_at();

-- Create trigger to increment use_count when template is used
CREATE OR REPLACE FUNCTION increment_template_use_count()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.workflow_definition->>'template_id' IS NOT NULL THEN
        UPDATE workflow_templates
        SET use_count = use_count + 1,
            last_used_at = CURRENT_TIMESTAMP
        WHERE template_id = NEW.workflow_definition->>'template_id';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER workflows_track_template_usage
    AFTER INSERT ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION increment_template_use_count();

