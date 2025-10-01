-- Final Test Data Generation Script for Automatos AI Platform
-- This script generates realistic test data without conflicting with existing data

-- 1. Insert Test Agents (only if they don't exist)
INSERT INTO agents (name, description, agent_type, status, configuration, performance_metrics, priority_level, max_concurrent_tasks, auto_start, created_at, updated_at, created_by) 
SELECT * FROM (VALUES
('Code Analyzer Agent', 'Specializes in analyzing code quality, patterns, and suggesting improvements', 'analysis', 'active', 
 '{"model": "gpt-4", "temperature": 0.3, "max_tokens": 2000, "tools": ["code_analysis", "pattern_detection"]}'::json, 
 '{"success_rate": 0.92, "avg_response_time": 2.3, "total_tasks": 156}'::json, 'high', 5, true, NOW() - INTERVAL '30 days', NOW(), 'system'),
('Document Processor', 'Handles document parsing, chunking, and metadata extraction', 'processing', 'active',
 '{"model": "gpt-3.5-turbo", "temperature": 0.1, "chunk_size": 512, "overlap": 50}'::json,
 '{"success_rate": 0.88, "avg_response_time": 4.1, "total_tasks": 89}'::json, 'medium', 3, true, NOW() - INTERVAL '25 days', NOW(), 'admin'),
('Context Optimizer', 'Optimizes context windows and reduces token usage', 'optimization', 'active',
 '{"model": "gpt-4", "temperature": 0.2, "compression_target": 0.7}'::json,
 '{"success_rate": 0.95, "avg_response_time": 1.8, "total_tasks": 234}'::json, 'high', 8, true, NOW() - INTERVAL '20 days', NOW(), 'system'),
('Memory Manager', 'Manages long-term memory storage and retrieval', 'memory', 'active',
 '{"model": "gpt-4", "temperature": 0.1, "memory_types": ["episodic", "semantic", "procedural"]}'::json,
 '{"success_rate": 0.90, "avg_response_time": 3.2, "total_tasks": 167}'::json, 'medium', 4, true, NOW() - INTERVAL '15 days', NOW(), 'admin'),
('Workflow Coordinator', 'Coordinates multi-agent workflows and task delegation', 'coordination', 'active',
 '{"model": "gpt-4", "temperature": 0.4, "coordination_strategies": ["sequential", "parallel", "conditional"]}'::json,
 '{"success_rate": 0.87, "avg_response_time": 5.6, "total_tasks": 78}'::json, 'high', 2, true, NOW() - INTERVAL '10 days', NOW(), 'system')
) AS t(name, description, agent_type, status, configuration, performance_metrics, priority_level, max_concurrent_tasks, auto_start, created_at, updated_at, created_by)
WHERE NOT EXISTS (SELECT 1 FROM agents WHERE agents.name = t.name);

-- 2. Insert Test Tasks (with NULL created_by to avoid foreign key issues)
INSERT INTO tasks (title, description, status, priority, assigned_to, created_by, due_date, completed_at, task_data, created_at, updated_at) 
SELECT * FROM (VALUES
('Analyze React Component Performance', 'Review and optimize performance of dashboard components', 'completed', 'high', 1, NULL, NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day', 
 '{"component": "dashboard.tsx", "metrics": {"render_time": 120, "re_renders": 3}}'::json, NOW() - INTERVAL '3 days', NOW() - INTERVAL '1 day'),
('Process API Documentation', 'Extract and chunk API documentation for knowledge base', 'in_progress', 'medium', 2, NULL, NOW() + INTERVAL '2 days', NULL,
 '{"document_type": "api_docs", "total_pages": 45, "chunks_created": 23}'::json, NOW() - INTERVAL '1 day', NOW()),
('Optimize Context for Chatbot', 'Reduce token usage in chatbot responses', 'pending', 'high', 3, NULL, NOW() + INTERVAL '1 day', NULL,
 '{"current_tokens": 2500, "target_tokens": 1800, "context_type": "conversation"}'::json, NOW() - INTERVAL '6 hours', NOW()),
('Store User Interaction Patterns', 'Create memory items from user behavior analysis', 'completed', 'medium', 4, NULL, NOW() - INTERVAL '1 day', NOW() - INTERVAL '4 hours',
 '{"interaction_count": 156, "patterns_found": 12, "memory_items_created": 8}'::json, NOW() - INTERVAL '2 days', NOW() - INTERVAL '4 hours'),
('Coordinate Agent Collaboration', 'Set up multi-agent workflow for document processing', 'in_progress', 'high', 5, NULL, NOW() + INTERVAL '3 days', NULL,
 '{"agents_involved": [1, 2, 4], "workflow_type": "sequential", "estimated_duration": "2 hours"}'::json, NOW() - INTERVAL '4 hours', NOW())
) AS t(title, description, status, priority, assigned_to, created_by, due_date, completed_at, task_data, created_at, updated_at);

-- 3. Insert Test Documents
INSERT INTO documents (filename, file_type, file_size, upload_date, processed_date, status, chunk_count, metadata, created_by, tags, description, file_hash, original_filename, file_path, content_hash, doc_metadata) 
SELECT * FROM (VALUES
('api_reference_v2.1.pdf', 'pdf', 2048576, NOW() - INTERVAL '5 days', NOW() - INTERVAL '4 days', 'processed', 45,
 '{"author": "API Team", "version": "2.1", "sections": 12}'::jsonb, 'admin', ARRAY['api', 'documentation', 'reference'],
 'Complete API reference documentation for version 2.1', 'sha256:abc123def456', 'API_Reference_v2.1.pdf', '/uploads/docs/api_reference_v2.1.pdf', 'sha256:def789ghi012', '{"page_count": 45, "tables": 8, "code_examples": 23}'::json),
('user_manual.md', 'markdown', 512000, NOW() - INTERVAL '3 days', NOW() - INTERVAL '2 days', 'processed', 28,
 '{"author": "Documentation Team", "version": "1.3", "sections": 8}'::jsonb, 'admin', ARRAY['manual', 'user_guide', 'documentation'],
 'User manual for platform features and usage', 'sha256:ghi456jkl789', 'User_Manual_v1.3.md', '/uploads/docs/user_manual.md', 'sha256:jkl012mno345', '{"word_count": 15420, "images": 12, "links": 67}'::json),
('system_architecture.json', 'json', 128000, NOW() - INTERVAL '2 days', NOW() - INTERVAL '1 day', 'processed', 5,
 '{"author": "Architecture Team", "version": "3.0", "components": 15}'::jsonb, 'system', ARRAY['architecture', 'system', 'json'],
 'System architecture configuration and component definitions', 'sha256:mno789pqr012', 'System_Architecture_v3.0.json', '/uploads/config/system_architecture.json', 'sha256:pqr345stu678', '{"components": 15, "connections": 23, "dependencies": 8}'::json),
('performance_metrics.csv', 'csv', 64000, NOW() - INTERVAL '1 day', NOW() - INTERVAL '6 hours', 'processing', 3,
 '{"author": "Analytics Team", "date_range": "last_30_days", "metrics": 12}'::jsonb, 'analytics', ARRAY['performance', 'metrics', 'csv'],
 'Performance metrics data for the last 30 days', 'sha256:stu901vwx234', 'Performance_Metrics_30d.csv', '/uploads/data/performance_metrics.csv', 'sha256:vwx567yza890', '{"rows": 720, "columns": 12, "date_range": "30_days"}'::json),
('codebase_analysis.txt', 'txt', 256000, NOW() - INTERVAL '4 hours', NULL, 'pending', NULL,
 '{"author": "Code Analysis Bot", "analysis_type": "quality", "files_analyzed": 156}'::jsonb, 'system', ARRAY['code', 'analysis', 'quality'],
 'Automated codebase quality analysis report', 'sha256:yza123bcd456', 'Codebase_Analysis_Report.txt', '/uploads/reports/codebase_analysis.txt', NULL, '{"files_analyzed": 156, "issues_found": 23, "suggestions": 45}'::json)
) AS t(filename, file_type, file_size, upload_date, processed_date, status, chunk_count, metadata, created_by, tags, description, file_hash, original_filename, file_path, content_hash, doc_metadata)
WHERE NOT EXISTS (SELECT 1 FROM documents WHERE documents.filename = t.filename);

-- 4. Insert Test System Metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, recorded_at) 
SELECT * FROM (VALUES
('cpu_usage', 45.2, 'percent', NOW() - INTERVAL '1 hour'),
('memory_usage', 67.8, 'percent', NOW() - INTERVAL '1 hour'),
('disk_usage', 23.4, 'percent', NOW() - INTERVAL '1 hour'),
('active_agents', 4, 'count', NOW() - INTERVAL '1 hour'),
('pending_tasks', 8, 'count', NOW() - INTERVAL '1 hour'),
('api_requests_per_minute', 156, 'requests', NOW() - INTERVAL '1 hour'),
('average_response_time', 1.2, 'seconds', NOW() - INTERVAL '1 hour'),
('error_rate', 0.02, 'percent', NOW() - INTERVAL '1 hour'),
('cpu_usage', 42.1, 'percent', NOW() - INTERVAL '2 hours'),
('memory_usage', 65.3, 'percent', NOW() - INTERVAL '2 hours'),
('disk_usage', 23.1, 'percent', NOW() - INTERVAL '2 hours'),
('active_agents', 5, 'count', NOW() - INTERVAL '2 hours')
) AS t(metric_name, metric_value, metric_unit, recorded_at);

-- 5. Insert Test Context Optimization Metrics (only required fields, generated columns auto-calculated)
INSERT INTO context_optimization_metrics (original_tokens, optimized_tokens, optimization_type, pattern_used, execution_time, recorded_at) 
SELECT * FROM (VALUES
(2500, 1800, 'semantic_compression', 'entity_extraction', 0.8, NOW() - INTERVAL '2 hours'),
(3200, 2100, 'keyword_extraction', 'tf_idf', 1.2, NOW() - INTERVAL '3 hours'),
(1800, 1400, 'summarization', 'abstractive', 0.6, NOW() - INTERVAL '4 hours'),
(4500, 2800, 'semantic_compression', 'concept_mapping', 1.5, NOW() - INTERVAL '5 hours'),
(2200, 1650, 'keyword_extraction', 'named_entities', 0.9, NOW() - INTERVAL '6 hours'),
(3800, 2400, 'summarization', 'extractive', 1.1, NOW() - INTERVAL '7 hours'),
(2900, 2000, 'semantic_compression', 'relationship_extraction', 1.0, NOW() - INTERVAL '8 hours'),
(1600, 1200, 'keyword_extraction', 'topic_modeling', 0.7, NOW() - INTERVAL '9 hours')
) AS t(original_tokens, optimized_tokens, optimization_type, pattern_used, execution_time, recorded_at);

-- 6. Insert Test Memory Items
INSERT INTO memory_items (agent_id, content, memory_type, memory_level, importance, access_count, last_access, decay_rate, associations, metadata, created_at, success_rate, usage_in_solutions, average_retrieval_time) 
SELECT * FROM (VALUES
(1, 'React components should use useMemo for expensive calculations', 'procedural', 'long_term', 0.9, 23, NOW() - INTERVAL '2 hours', 0.05, ARRAY['react', 'performance', 'optimization'], '{"category": "best_practices", "confidence": 0.95}'::json, NOW() - INTERVAL '10 days', 0.87, 15, 0.3),
(2, 'PDF documents with tables require special parsing logic', 'episodic', 'medium_term', 0.8, 12, NOW() - INTERVAL '4 hours', 0.08, ARRAY['pdf', 'parsing', 'tables'], '{"category": "processing", "confidence": 0.88}'::json, NOW() - INTERVAL '8 days', 0.92, 8, 0.5),
(3, 'Context optimization works best with semantic similarity thresholds of 0.7', 'semantic', 'long_term', 0.95, 45, NOW() - INTERVAL '1 hour', 0.03, ARRAY['context', 'optimization', 'similarity'], '{"category": "parameters", "confidence": 0.98}'::json, NOW() - INTERVAL '15 days', 0.94, 32, 0.2),
(4, 'User prefers concise responses over detailed explanations', 'episodic', 'short_term', 0.6, 8, NOW() - INTERVAL '6 hours', 0.12, ARRAY['user_preference', 'communication'], '{"category": "user_behavior", "confidence": 0.75}'::json, NOW() - INTERVAL '3 days', 0.78, 5, 0.8),
(5, 'Sequential workflows perform better than parallel for dependent tasks', 'procedural', 'long_term', 0.85, 18, NOW() - INTERVAL '3 hours', 0.06, ARRAY['workflow', 'coordination', 'performance'], '{"category": "strategy", "confidence": 0.90}'::json, NOW() - INTERVAL '12 days', 0.89, 12, 0.4)
) AS t(agent_id, content, memory_type, memory_level, importance, access_count, last_access, decay_rate, associations, metadata, created_at, success_rate, usage_in_solutions, average_retrieval_time);

-- 7. Insert Test Knowledge Nodes
INSERT INTO knowledge_nodes (agent_id, concept, description, node_type, importance, confidence, metadata, created_at) 
SELECT * FROM (VALUES
(1, 'React Performance Optimization', 'Techniques for optimizing React component performance', 'technical_concept', 0.9, 0.95, '{"domain": "frontend", "complexity": "intermediate"}'::jsonb, NOW() - INTERVAL '10 days'),
(2, 'Document Parsing Strategies', 'Different approaches to parsing various document formats', 'methodology', 0.85, 0.88, '{"domain": "processing", "complexity": "advanced"}'::jsonb, NOW() - INTERVAL '8 days'),
(3, 'Context Compression Algorithms', 'Mathematical approaches to context compression', 'algorithm', 0.95, 0.92, '{"domain": "optimization", "complexity": "expert"}'::jsonb, NOW() - INTERVAL '15 days'),
(4, 'Memory Retrieval Patterns', 'Common patterns in memory retrieval and association', 'pattern', 0.8, 0.85, '{"domain": "cognitive", "complexity": "intermediate"}'::jsonb, NOW() - INTERVAL '12 days'),
(5, 'Workflow Coordination Principles', 'Fundamental principles for coordinating multi-agent workflows', 'principle', 0.9, 0.90, '{"domain": "coordination", "complexity": "advanced"}'::jsonb, NOW() - INTERVAL '7 days'),
(NULL, 'User Interaction Patterns', 'Common patterns in how users interact with AI systems', 'collective_knowledge', 0.75, 0.80, '{"domain": "ux", "complexity": "beginner"}'::jsonb, NOW() - INTERVAL '5 days'),
(NULL, 'Error Handling Best Practices', 'Best practices for handling errors in distributed systems', 'collective_knowledge', 0.85, 0.88, '{"domain": "reliability", "complexity": "intermediate"}'::jsonb, NOW() - INTERVAL '6 days'),
(1, 'Code Quality Metrics', 'Quantitative measures of code quality and maintainability', 'metric', 0.8, 0.87, '{"domain": "quality", "complexity": "intermediate"}'::jsonb, NOW() - INTERVAL '9 days')
) AS t(agent_id, concept, description, node_type, importance, confidence, metadata, created_at);

-- Display summary
SELECT 'Final test data generation completed!' as status;
SELECT 'Agents: ' || COUNT(*) as agents_count FROM agents;
SELECT 'Tasks: ' || COUNT(*) as tasks_count FROM tasks;
SELECT 'Documents: ' || COUNT(*) as documents_count FROM documents;
SELECT 'System Metrics: ' || COUNT(*) as metrics_count FROM system_metrics;
SELECT 'Context Optimization Metrics: ' || COUNT(*) as context_metrics_count FROM context_optimization_metrics;
SELECT 'Memory Items: ' || COUNT(*) as memory_count FROM memory_items;
SELECT 'Knowledge Nodes: ' || COUNT(*) as knowledge_count FROM knowledge_nodes;
