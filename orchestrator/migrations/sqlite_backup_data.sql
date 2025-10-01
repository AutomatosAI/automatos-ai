-- SQLite Data Backup - Generated before PostgreSQL migration
-- Date: 2025-09-16

=== agents ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE agents (
	id INTEGER NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	description TEXT, 
	agent_type VARCHAR(100) NOT NULL, 
	status VARCHAR(50), 
	configuration JSON, 
	performance_metrics JSON, 
	priority_level VARCHAR(50), 
	max_concurrent_tasks INTEGER, 
	auto_start BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	created_by VARCHAR(255), quality_score REAL, emergence_score REAL, performance REAL, reliability REAL, readiness REAL, coherence REAL, efficiency REAL, eci REAL, validity REAL, discriminatory_power REAL, 
	PRIMARY KEY (id)
);
INSERT INTO agents VALUES(1,'Test Agent','A test agent for API testing','custom','active','{"model": "test"}',NULL,'medium',5,0,'2025-09-16 19:44:23','2025-09-16 19:44:23','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(2,'<script>alert("xss")</script>','''; DROP TABLE users; --','custom','active','{"model": "test"}',NULL,'medium',5,0,'2025-09-16 19:44:23','2025-09-16 19:44:23','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(3,'Research Assistant','Specialized in information gathering and research tasks','data_analyst','active','{"skills": ["research", "analysis", "data_collection"], "capabilities": {"max_tokens": 4000, "temperature": 0.7, "model": "gpt-4"}, "role": "researcher"}',NULL,'medium',5,0,'2025-09-16 21:02:47','2025-09-16 21:02:47','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(4,'Research Assistant 1758056600','Specialized in information gathering and research tasks','data_analyst','active','{"skills": ["research", "analysis", "data_collection"], "capabilities": {"max_tokens": 4000, "temperature": 0.7, "model": "gpt-4"}, "role": "researcher"}',NULL,'medium',5,0,'2025-09-16 21:03:20','2025-09-16 21:03:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(5,'Content Creator 1758056600','Expert in content creation and document writing','custom','active','{"skills": ["writing", "editing", "content_creation"], "capabilities": {"max_tokens": 3000, "temperature": 0.8, "model": "gpt-3.5-turbo"}, "role": "content_creator"}',NULL,'medium',5,0,'2025-09-16 21:03:20','2025-09-16 21:03:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(6,'Data Analyst 1758056600','Specialized in data analysis and visualization','data_analyst','active','{"skills": ["data_analysis", "statistics", "visualization"], "capabilities": {"max_tokens": 2000, "temperature": 0.3, "model": "gpt-4"}, "role": "analyst"}',NULL,'medium',5,0,'2025-09-16 21:03:20','2025-09-16 21:03:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(7,'Workflow Manager 1758056600','Manages and orchestrates complex workflows','system','active','{"skills": ["workflow_management", "orchestration", "automation"], "capabilities": {"max_tokens": 3500, "temperature": 0.5, "model": "gpt-4"}, "role": "workflow_manager"}',NULL,'medium',5,0,'2025-09-16 21:03:21','2025-09-16 21:03:21','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(8,'Security Expert 1758056600','Specialized in security analysis and audits','security_expert','active','{"skills": ["security_analysis", "vulnerability_assessment", "compliance"], "capabilities": {"max_tokens": 3000, "temperature": 0.4, "model": "gpt-4"}, "role": "security_specialist"}',NULL,'medium',5,0,'2025-09-16 21:03:21','2025-09-16 21:03:21','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(9,'Research Assistant 1758056626','Specialized in information gathering and research tasks','data_analyst','active','{"skills": ["research", "analysis", "data_collection"], "capabilities": {"max_tokens": 4000, "temperature": 0.7, "model": "gpt-4"}, "role": "researcher"}',NULL,'medium',5,0,'2025-09-16 21:03:46','2025-09-16 21:03:46','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(10,'Content Creator 1758056626','Expert in content creation and document writing','custom','active','{"skills": ["writing", "editing", "content_creation"], "capabilities": {"max_tokens": 3000, "temperature": 0.8, "model": "gpt-3.5-turbo"}, "role": "content_creator"}',NULL,'medium',5,0,'2025-09-16 21:03:46','2025-09-16 21:03:46','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(11,'Data Analyst 1758056626','Specialized in data analysis and visualization','data_analyst','active','{"skills": ["data_analysis", "statistics", "visualization"], "capabilities": {"max_tokens": 2000, "temperature": 0.3, "model": "gpt-4"}, "role": "analyst"}',NULL,'medium',5,0,'2025-09-16 21:03:46','2025-09-16 21:03:46','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(12,'Workflow Manager 1758056626','Manages and orchestrates complex workflows','system','active','{"skills": ["workflow_management", "orchestration", "automation"], "capabilities": {"max_tokens": 3500, "temperature": 0.5, "model": "gpt-4"}, "role": "workflow_manager"}',NULL,'medium',5,0,'2025-09-16 21:03:46','2025-09-16 21:03:46','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(13,'Security Expert 1758056626','Specialized in security analysis and audits','security_expert','active','{"skills": ["security_analysis", "vulnerability_assessment", "compliance"], "capabilities": {"max_tokens": 3000, "temperature": 0.4, "model": "gpt-4"}, "role": "security_specialist"}',NULL,'medium',5,0,'2025-09-16 21:03:46','2025-09-16 21:03:46','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(14,'Research Assistant 1758056648','Specialized in information gathering and research tasks','data_analyst','active','{"skills": ["research", "analysis", "data_collection"], "capabilities": {"max_tokens": 4000, "temperature": 0.7, "model": "gpt-4"}, "role": "researcher"}',NULL,'medium',5,0,'2025-09-16 21:04:08','2025-09-16 21:04:08','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(15,'Content Creator 1758056648','Expert in content creation and document writing','custom','active','{"skills": ["writing", "editing", "content_creation"], "capabilities": {"max_tokens": 3000, "temperature": 0.8, "model": "gpt-3.5-turbo"}, "role": "content_creator"}',NULL,'medium',5,0,'2025-09-16 21:04:08','2025-09-16 21:04:08','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(16,'Data Analyst 1758056648','Specialized in data analysis and visualization','data_analyst','active','{"skills": ["data_analysis", "statistics", "visualization"], "capabilities": {"max_tokens": 2000, "temperature": 0.3, "model": "gpt-4"}, "role": "analyst"}',NULL,'medium',5,0,'2025-09-16 21:04:08','2025-09-16 21:04:08','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(17,'Workflow Manager 1758056648','Manages and orchestrates complex workflows','system','active','{"skills": ["workflow_management", "orchestration", "automation"], "capabilities": {"max_tokens": 3500, "temperature": 0.5, "model": "gpt-4"}, "role": "workflow_manager"}',NULL,'medium',5,0,'2025-09-16 21:04:08','2025-09-16 21:04:08','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(18,'Security Expert 1758056648','Specialized in security analysis and audits','security_expert','active','{"skills": ["security_analysis", "vulnerability_assessment", "compliance"], "capabilities": {"max_tokens": 3000, "temperature": 0.4, "model": "gpt-4"}, "role": "security_specialist"}',NULL,'medium',5,0,'2025-09-16 21:04:08','2025-09-16 21:04:08','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(19,'Research Assistant 1758056660','Specialized in information gathering and research tasks','data_analyst','active','{"skills": ["research", "analysis", "data_collection"], "capabilities": {"max_tokens": 4000, "temperature": 0.7, "model": "gpt-4"}, "role": "researcher"}',NULL,'medium',5,0,'2025-09-16 21:04:20','2025-09-16 21:04:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(20,'Content Creator 1758056660','Expert in content creation and document writing','custom','active','{"skills": ["writing", "editing", "content_creation"], "capabilities": {"max_tokens": 3000, "temperature": 0.8, "model": "gpt-3.5-turbo"}, "role": "content_creator"}',NULL,'medium',5,0,'2025-09-16 21:04:20','2025-09-16 21:04:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(21,'Data Analyst 1758056660','Specialized in data analysis and visualization','data_analyst','active','{"skills": ["data_analysis", "statistics", "visualization"], "capabilities": {"max_tokens": 2000, "temperature": 0.3, "model": "gpt-4"}, "role": "analyst"}',NULL,'medium',5,0,'2025-09-16 21:04:20','2025-09-16 21:04:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(22,'Workflow Manager 1758056660','Manages and orchestrates complex workflows','system','active','{"skills": ["workflow_management", "orchestration", "automation"], "capabilities": {"max_tokens": 3500, "temperature": 0.5, "model": "gpt-4"}, "role": "workflow_manager"}',NULL,'medium',5,0,'2025-09-16 21:04:20','2025-09-16 21:04:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(23,'Security Expert 1758056660','Specialized in security analysis and audits','security_expert','active','{"skills": ["security_analysis", "vulnerability_assessment", "compliance"], "capabilities": {"max_tokens": 3000, "temperature": 0.4, "model": "gpt-4"}, "role": "security_specialist"}',NULL,'medium',5,0,'2025-09-16 21:04:20','2025-09-16 21:04:20','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(24,'Journey Test Agent','Agent created during user journey testing','custom','active','{"skills": ["testing", "validation", "journey_testing"], "capabilities": {"max_tokens": 2000, "temperature": 0.6, "model": "gpt-3.5-turbo"}, "role": "journey_tester"}',NULL,'medium',5,0,'2025-09-16 21:05:26','2025-09-16 21:05:26','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(25,'Integration Test Agent','Specialized agent for integrated workflow testing','specialized','active','{"skills": ["document_analysis", "content_extraction", "integration_testing"], "capabilities": {"max_tokens": 4000, "temperature": 0.5, "model": "gpt-4"}, "role": "integration_specialist"}',NULL,'medium',5,0,'2025-09-16 21:05:26','2025-09-16 21:05:26','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(26,'Journey Test Agent 1758056806','Updated description for journey testing','custom','active','{"skills": ["testing", "validation", "journey_testing"], "capabilities": {"max_tokens": 2000, "temperature": 0.6, "model": "gpt-3.5-turbo"}, "role": "journey_tester"}',NULL,'medium',5,0,'2025-09-16 21:06:46','2025-09-16 21:06:46','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(27,'Integration Test Agent 1758056807','Specialized agent for integrated workflow testing','specialized','active','{"skills": ["document_analysis", "content_extraction", "integration_testing"], "capabilities": {"max_tokens": 4000, "temperature": 0.5, "model": "gpt-4"}, "role": "integration_specialist"}',NULL,'medium',5,0,'2025-09-16 21:06:47','2025-09-16 21:06:47','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(28,'Journey Test Agent 1758056869','Updated description for journey testing','custom','inactive','{"skills": ["testing", "validation", "journey_testing"], "capabilities": {"max_tokens": 2000, "temperature": 0.6, "model": "gpt-3.5-turbo"}, "role": "journey_tester"}',NULL,'medium',5,0,'2025-09-16 21:07:49','2025-09-16 21:07:49','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(29,'Integration Test Agent 1758056869','Specialized agent for integrated workflow testing','specialized','active','{"skills": ["document_analysis", "content_extraction", "integration_testing"], "capabilities": {"max_tokens": 4000, "temperature": 0.5, "model": "gpt-4"}, "role": "integration_specialist"}',NULL,'medium',5,0,'2025-09-16 21:07:49','2025-09-16 21:07:49','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(30,'Journey Test Agent 1758056886','Updated description for journey testing','custom','inactive','{"skills": ["testing", "validation", "journey_testing"], "capabilities": {"max_tokens": 2000, "temperature": 0.6, "model": "gpt-3.5-turbo"}, "role": "journey_tester"}',NULL,'medium',5,0,'2025-09-16 21:08:06','2025-09-16 21:08:06','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO agents VALUES(31,'Integration Test Agent 1758056887','Specialized agent for integrated workflow testing','specialized','active','{"skills": ["document_analysis", "content_extraction", "integration_testing"], "capabilities": {"max_tokens": 4000, "temperature": 0.5, "model": "gpt-4"}, "role": "integration_specialist"}',NULL,'medium',5,0,'2025-09-16 21:08:07','2025-09-16 21:08:07','api',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
COMMIT;

=== skills ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE skills (
	id INTEGER NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	description TEXT, 
	skill_type VARCHAR(100) NOT NULL, 
	category VARCHAR(100) NOT NULL, 
	implementation TEXT, 
	parameters JSON, 
	performance_data JSON, 
	is_active BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	created_by VARCHAR(255), 
	PRIMARY KEY (id)
);
COMMIT;

=== patterns ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE patterns (
	id INTEGER NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	description TEXT, 
	pattern_type VARCHAR(100) NOT NULL, 
	pattern_data JSON, 
	usage_count INTEGER, 
	effectiveness_score FLOAT, 
	is_active BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	created_by VARCHAR(255), 
	PRIMARY KEY (id)
);
COMMIT;

=== workflows ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE workflows (
	id INTEGER NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	description TEXT, 
	workflow_definition JSON, 
	status VARCHAR(50), 
	created_at DATETIME, 
	updated_at DATETIME, 
	created_by VARCHAR(255), owner VARCHAR(255), tags JSON, default_policy_id VARCHAR(128), priority VARCHAR(50), expected_duration INTEGER, complexity_score REAL, success_rate REAL, 
	PRIMARY KEY (id)
);
INSERT INTO workflows VALUES(1,'Test Workflow Schema Fix','Testing if the database schema issues are resolved','{"category": "testing", "priority": "medium", "config": {"timeout": 300, "retry_count": 3}, "steps": [{"name": "Initialize", "type": "setup", "config": {}}, {"name": "Process", "type": "execution", "config": {}}], "agents": [], "version": "1.0"}','draft','2025-09-16 21:21:45','2025-09-16 21:21:45','test-system','system','["test", "schema-fix"]','test-policy-123',NULL,NULL,NULL,NULL);
INSERT INTO workflows VALUES(2,'Document Analysis Orchestration','Complex workflow demonstrating agent coordination for document analysis','{"category": "analysis", "priority": "high", "config": {"timeout": 600, "retry_count": 2, "parallel_processing": true}, "steps": [{"name": "Document Preprocessing", "type": "preprocessing", "agent_type": "data_analyst", "config": {"extract_metadata": true}}, {"name": "Content Analysis", "type": "analysis", "agent_type": "code_architect", "config": {"depth": "comprehensive"}}, {"name": "Security Review", "type": "security", "agent_type": "security_expert", "config": {"scan_level": "thorough"}}, {"name": "Report Generation", "type": "reporting", "agent_type": "data_analyst", "config": {"format": "detailed"}}], "agents": [], "version": "1.0"}','draft','2025-09-16 21:24:36','2025-09-16 21:24:36','orchestrator-system','orchestrator-test','["document-analysis", "multi-agent", "orchestration"]','analysis-policy-001',NULL,NULL,NULL,NULL);
INSERT INTO workflows VALUES(3,'Orchestration Test Workflow 1758057951','Comprehensive test workflow for orchestration validation','{"category": "testing", "priority": "high", "config": {"timeout": 300, "retry_count": 3, "parallel_processing": true, "enable_monitoring": true}, "steps": [{"name": "Initialize", "type": "setup", "agent_type": "system"}, {"name": "Analyze", "type": "analysis", "agent_type": "data_analyst"}, {"name": "Process", "type": "processing", "agent_type": "code_architect"}, {"name": "Validate", "type": "validation", "agent_type": "security_expert"}, {"name": "Report", "type": "reporting", "agent_type": "data_analyst"}], "agents": [], "version": "1.0"}','draft','2025-09-16 21:25:51','2025-09-16 21:25:51','orchestration-test-suite','test-orchestrator','["orchestration", "testing", "validation", "comprehensive"]','test-policy-comprehensive',NULL,NULL,NULL,NULL);
INSERT INTO workflows VALUES(4,'Advanced Multi-Agent Research Pipeline','Sophisticated workflow demonstrating full orchestration capabilities with multiple specialized agents','{"category": "research", "priority": "critical", "config": {"timeout": 900, "retry_count": 3, "parallel_processing": true, "enable_monitoring": true, "quality_threshold": 0.95}, "steps": [{"name": "Research Planning", "type": "planning", "agent_type": "data_analyst", "config": {"depth": "comprehensive", "sources": "multiple"}}, {"name": "Data Collection", "type": "collection", "agent_type": "code_architect", "config": {"parallel": true, "validation": true}}, {"name": "Security Analysis", "type": "security", "agent_type": "security_expert", "config": {"scan_level": "deep", "compliance_check": true}}, {"name": "Performance Optimization", "type": "optimization", "agent_type": "performance_optimizer", "config": {"metrics": "comprehensive", "benchmarking": true}}, {"name": "Infrastructure Assessment", "type": "infrastructure", "agent_type": "infrastructure_manager", "config": {"scalability": true, "reliability": true}}, {"name": "Final Report Generation", "type": "reporting", "agent_type": "data_analyst", "config": {"format": "executive", "visualizations": true}}], "agents": [], "version": "1.0"}','draft','2025-09-16 21:27:14','2025-09-16 21:27:14','advanced-orchestration-demo','orchestrator-demo','["research", "multi-agent", "pipeline", "advanced", "demo"]','research-pipeline-policy',NULL,NULL,NULL,NULL);
COMMIT;

=== documents ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE documents (
	id INTEGER NOT NULL, 
	filename VARCHAR(255) NOT NULL, 
	original_filename VARCHAR(255), 
	file_type VARCHAR(100), 
	file_size INTEGER, 
	file_path VARCHAR(500), 
	content_hash VARCHAR(255), 
	status VARCHAR(50), 
	chunk_count INTEGER, 
	tags JSON, 
	description TEXT, 
	doc_metadata JSON, 
	upload_date DATETIME, 
	processed_date DATETIME, 
	created_by VARCHAR(255), 
	PRIMARY KEY (id)
);
INSERT INTO documents VALUES(1,'sample_report.pdf','sample_report.pdf','pdf',458,'/tmp/automotas_uploads/d437fd146ac15f41a3eb38fd7a94b4e1c434615b1e618d154bea39fcf7366b16_sample_report.pdf','d437fd146ac15f41a3eb38fd7a94b4e1c434615b1e618d154bea39fcf7366b16','failed',0,'[]','Test document: sample_report.pdf uploaded by test system',NULL,'2025-09-16 21:03:46',NULL,'system');
INSERT INTO documents VALUES(2,'project_requirements.txt','project_requirements.txt','text',613,'/tmp/automotas_uploads/30f94f2055e35e9d899411a6cd077443467aba6126bb3a3bace66a3041cbc3b6_project_requirements.txt','30f94f2055e35e9d899411a6cd077443467aba6126bb3a3bace66a3041cbc3b6','failed',0,'[]','Test document: project_requirements.txt uploaded by test system',NULL,'2025-09-16 21:03:46',NULL,'system');
INSERT INTO documents VALUES(3,'technical_specification.md','technical_specification.md','text',623,'/tmp/automotas_uploads/cdd1c427039c2dcc73825a880600839b7d39a995440c2f6820c23414b87382a1_technical_specification.md','cdd1c427039c2dcc73825a880600839b7d39a995440c2f6820c23414b87382a1','failed',0,'[]','Test document: technical_specification.md uploaded by test system',NULL,'2025-09-16 21:03:47',NULL,'system');
INSERT INTO documents VALUES(4,'user_manual.txt','user_manual.txt','text',538,'/tmp/automotas_uploads/36dfaa51d787561f813b3faa19c543650a3789d08362986f70dfd29e1f8ffcb2_user_manual.txt','36dfaa51d787561f813b3faa19c543650a3789d08362986f70dfd29e1f8ffcb2','failed',0,'[]','Test document: user_manual.txt uploaded by test system',NULL,'2025-09-16 21:03:47',NULL,'system');
INSERT INTO documents VALUES(6,'integration_test_document.txt','integration_test_document.txt','text',765,'/tmp/automotas_uploads/883bceb0e24c908f1fdc99a79a10bdf70d82ca912b27327c8c3b4a53382436d7_integration_test_document.txt','883bceb0e24c908f1fdc99a79a10bdf70d82ca912b27327c8c3b4a53382436d7','failed',0,'[]','Integration test document for workflow testing',NULL,'2025-09-16 21:05:26',NULL,'system');
COMMIT;

=== system_configurations ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE system_configurations (
	id INTEGER NOT NULL, 
	config_key VARCHAR(255) NOT NULL, 
	config_value JSON, 
	description TEXT, 
	is_active BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	updated_by VARCHAR(255), 
	PRIMARY KEY (id), 
	UNIQUE (config_key)
);
INSERT INTO system_configurations VALUES(1,'system.max_agents','{"value": 100}','Maximum number of agents allowed in the system',1,'2025-09-04 16:23:34','2025-09-04 16:23:34',NULL);
INSERT INTO system_configurations VALUES(2,'system.default_timeout','{"value": 300}','Default timeout for agent operations (seconds)',1,'2025-09-04 16:23:34','2025-09-04 16:23:34',NULL);
INSERT INTO system_configurations VALUES(3,'rag.default_model','{"value": "sentence-transformers/all-MiniLM-L6-v2"}','Default embedding model for RAG system',1,'2025-09-04 16:23:34','2025-09-04 16:23:34',NULL);
INSERT INTO system_configurations VALUES(4,'workflow.max_concurrent','{"value": 10}','Maximum concurrent workflow executions',1,'2025-09-04 16:23:34','2025-09-04 16:23:34',NULL);
COMMIT;

=== rag_configurations ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE rag_configurations (
	id INTEGER NOT NULL, 
	name VARCHAR(255) NOT NULL, 
	embedding_model VARCHAR(255), 
	chunk_size INTEGER, 
	chunk_overlap INTEGER, 
	retrieval_strategy VARCHAR(100), 
	top_k INTEGER, 
	similarity_threshold FLOAT, 
	configuration JSON, 
	is_active BOOLEAN, 
	created_at DATETIME, 
	updated_at DATETIME, 
	created_by VARCHAR(255), 
	PRIMARY KEY (id)
);
INSERT INTO rag_configurations VALUES(1,'default','sentence-transformers/all-MiniLM-L6-v2',1000,200,'similarity',5,0.69999999999999995559,'{"max_tokens": 4000, "temperature": 0.7, "use_reranking": true}',1,'2025-09-04 16:23:34','2025-09-04 16:23:34','system');
INSERT INTO rag_configurations VALUES(2,'Test Item','sentence-transformers/all-MiniLM-L6-v2',1000,200,'similarity',5,0.69999999999999995559,'{}',1,'2025-09-16 19:44:24','2025-09-16 19:44:24','system');
INSERT INTO rag_configurations VALUES(3,'Test Item','sentence-transformers/all-MiniLM-L6-v2',1000,200,'similarity',5,0.69999999999999995559,'{}',1,'2025-09-16 19:44:24','2025-09-16 19:44:24','system');
INSERT INTO rag_configurations VALUES(4,'<script>alert("xss")</script>','sentence-transformers/all-MiniLM-L6-v2',1000,200,'similarity',5,0.69999999999999995559,'{}',1,'2025-09-16 19:44:24','2025-09-16 19:44:24','system');
INSERT INTO rag_configurations VALUES(5,'Test Item','sentence-transformers/all-MiniLM-L6-v2',1000,200,'similarity',5,0.69999999999999995559,'{}',1,'2025-09-16 19:44:24','2025-09-16 19:44:24','system');
COMMIT;

=== agent_skills ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE agent_skills (
	agent_id INTEGER, 
	skill_id INTEGER, 
	FOREIGN KEY(agent_id) REFERENCES agents (id), 
	FOREIGN KEY(skill_id) REFERENCES skills (id)
);
COMMIT;

=== workflow_agents ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE workflow_agents (
	workflow_id INTEGER, 
	agent_id INTEGER, 
	FOREIGN KEY(workflow_id) REFERENCES workflows (id), 
	FOREIGN KEY(agent_id) REFERENCES agents (id)
);
COMMIT;

=== workflow_executions ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE workflow_executions (
	id INTEGER NOT NULL, 
	workflow_id INTEGER, 
	agent_id INTEGER, 
	status VARCHAR(50), 
	input_data JSON, 
	output_data JSON, 
	execution_log TEXT, 
	started_at DATETIME, 
	completed_at DATETIME, 
	error_message TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(workflow_id) REFERENCES workflows (id), 
	FOREIGN KEY(agent_id) REFERENCES agents (id)
);
INSERT INTO workflow_executions VALUES(1,1,1,'completed','{"task": "Test orchestration functionality", "type": "system_test", "priority": "medium"}','{"result": "Workflow completed successfully", "steps_completed": 6, "execution_time": "6.099999999999999s", "agent_performance": "excellent", "generated_artifacts": ["report.pdf", "analysis.json", "summary.md"]}','Workflow executed successfully in 6.099999999999999s with 6 steps completed.','2025-09-16 21:23:51','2025-09-16 21:23:57.678281',NULL);
INSERT INTO workflow_executions VALUES(2,2,1,'completed','{"document_id": "test-doc-001", "analysis_type": "comprehensive", "priority": "high", "requirements": {"security_scan": true, "performance_analysis": true, "code_review": true, "documentation_check": true}}','{"result": "Workflow completed successfully", "steps_completed": 6, "execution_time": "6.099999999999999s", "agent_performance": "excellent", "generated_artifacts": ["report.pdf", "analysis.json", "summary.md"]}','Workflow executed successfully in 6.099999999999999s with 6 steps completed.','2025-09-16 21:24:42','2025-09-16 21:24:48.926910',NULL);
INSERT INTO workflow_executions VALUES(3,3,1,'completed','{"task": "Comprehensive orchestration test", "type": "validation", "priority": "high", "test_parameters": {"validate_schema": true, "test_agents": true, "monitor_progress": true, "check_coordination": true}}','{"result": "Workflow completed successfully", "steps_completed": 6, "execution_time": "6.099999999999999s", "agent_performance": "excellent", "generated_artifacts": ["report.pdf", "analysis.json", "summary.md"]}','Workflow executed successfully in 6.099999999999999s with 6 steps completed.','2025-09-16 21:25:51','2025-09-16 21:25:57.503469',NULL);
INSERT INTO workflow_executions VALUES(4,4,1,'completed','{"research_topic": "AI Orchestration System Performance Analysis", "scope": "comprehensive", "deliverables": ["technical_report", "performance_metrics", "security_assessment", "scalability_analysis"], "stakeholders": ["engineering_team", "security_team", "management"], "deadline": "2025-09-17T00:00:00Z", "quality_requirements": {"accuracy": 0.95, "completeness": 0.98, "security_compliance": true, "performance_benchmarks": true}}','{"result": "Workflow completed successfully", "steps_completed": 6, "execution_time": "6.099999999999999s", "agent_performance": "excellent", "generated_artifacts": ["report.pdf", "analysis.json", "summary.md"]}','Workflow executed successfully in 6.099999999999999s with 6 steps completed.','2025-09-16 21:27:26','2025-09-16 21:27:33.073175',NULL);
COMMIT;

=== tasks ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE tasks (
	id INTEGER NOT NULL, 
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
	field_timestamp DATETIME, 
	propagation_timestamp DATETIME, 
	interactions JSON, 
	emergent_effect FLOAT, 
	embeddings JSON, 
	stability FLOAT, 
	prev_field_value FLOAT, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id)
);
COMMIT;

=== users ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE users (
	id INTEGER NOT NULL, 
	username VARCHAR(255) NOT NULL, 
	email VARCHAR(255) NOT NULL, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (username), 
	UNIQUE (email)
);
COMMIT;

=== memory_items ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE memory_items (
	id VARCHAR(255) NOT NULL, 
	session_id VARCHAR(255) NOT NULL, 
	content JSON NOT NULL, 
	memory_type VARCHAR(50) NOT NULL, 
	memory_level VARCHAR(50) NOT NULL, 
	importance FLOAT, 
	access_count INTEGER, 
	decay_factor FLOAT, 
	consolidation_score FLOAT, 
	tags JSON, 
	created_at DATETIME, 
	last_access DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id)
);
COMMIT;

=== external_knowledge ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE external_knowledge (
	id INTEGER NOT NULL, 
	content JSON NOT NULL, 
	source VARCHAR(255) NOT NULL, 
	knowledge_metadata JSON, 
	access_count INTEGER, 
	created_at DATETIME, 
	updated_at DATETIME, 
	PRIMARY KEY (id)
);
COMMIT;

=== evaluation_results ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE evaluation_results (
	id INTEGER NOT NULL, 
	evaluation_id VARCHAR(255) NOT NULL, 
	evaluation_type VARCHAR(100) NOT NULL, 
	scope VARCHAR(100) NOT NULL, 
	target_id VARCHAR(255) NOT NULL, 
	overall_score FLOAT NOT NULL, 
	detailed_results JSON, 
	success BOOLEAN, 
	error_message TEXT, 
	execution_time_seconds FLOAT, 
	user_id INTEGER, 
	created_at DATETIME, 
	PRIMARY KEY (id), 
	UNIQUE (evaluation_id)
);
COMMIT;

=== benchmark_assessments ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE benchmark_assessments (
	id INTEGER NOT NULL, 
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
	created_at DATETIME, 
	PRIMARY KEY (id)
);
COMMIT;

=== component_metrics ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE component_metrics (
	id INTEGER NOT NULL, 
	component_id VARCHAR(255) NOT NULL, 
	component_type VARCHAR(100) NOT NULL, 
	performance_score FLOAT, 
	reliability_score FLOAT, 
	readiness_score FLOAT, 
	capability_rating FLOAT, 
	complexity_index FLOAT, 
	environment_factor FLOAT, 
	assessment_details JSON, 
	assessment_timestamp DATETIME, 
	PRIMARY KEY (id)
);
COMMIT;

=== integration_analyses ===
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE integration_analyses (
	id INTEGER NOT NULL, 
	system_id VARCHAR(255) NOT NULL, 
	coherence_score FLOAT, 
	efficiency_score FLOAT, 
	emergence_score FLOAT, 
	integration_score FLOAT, 
	integration_classification VARCHAR(50), 
	analysis_data JSON, 
	recommendations JSON, 
	confidence_level FLOAT, 
	created_at DATETIME, 
	PRIMARY KEY (id)
);
COMMIT;

