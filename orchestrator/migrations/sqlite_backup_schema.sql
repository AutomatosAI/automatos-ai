-- SQLite Schema Backup - Generated before PostgreSQL migration
-- Date: 2025-09-16

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
CREATE TABLE agent_skills (
	agent_id INTEGER, 
	skill_id INTEGER, 
	FOREIGN KEY(agent_id) REFERENCES agents (id), 
	FOREIGN KEY(skill_id) REFERENCES skills (id)
);
CREATE TABLE workflow_agents (
	workflow_id INTEGER, 
	agent_id INTEGER, 
	FOREIGN KEY(workflow_id) REFERENCES workflows (id), 
	FOREIGN KEY(agent_id) REFERENCES agents (id)
);
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
