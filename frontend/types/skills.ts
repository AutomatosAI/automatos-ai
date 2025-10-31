
/**
 * TypeScript Types for PRD-22 Skills System
 * =========================================
 * 
 * Type definitions for skills, sources, and related entities.
 */

export enum SkillSourceType {
  GIT = 'git',
  UPLOAD = 'upload',
  SEED = 'seed',
  MARKETPLACE = 'marketplace'
}

export enum SkillSourceStatus {
  ACTIVE = 'active',
  ERROR = 'error',
  SYNCING = 'syncing',
  INACTIVE = 'inactive'
}

export enum SkillFileType {
  CORE = 'core',
  RESOURCE = 'resource',
  SCRIPT = 'script',
  EXAMPLE = 'example'
}

export enum SkillLoadLevel {
  METADATA = 1,
  CORE = 2,
  RESOURCE = 3
}

export interface SkillSource {
  id: number;
  source_name: string;
  source_type: SkillSourceType;
  git_url?: string;
  git_branch?: string;
  git_commit_sha?: string;
  local_cache_path: string;
  auto_update: boolean;
  skills_discovered: number;
  status: SkillSourceStatus;
  last_sync_at?: string;
  last_sync_status?: string;
  last_sync_error?: string;
  created_at: string;
  updated_at: string;
}

export interface SkillFile {
  file_path: string;
  file_type: SkillFileType;
  content_summary?: string;
  file_size_bytes?: number;
  estimated_tokens?: number;
  load_level: SkillLoadLevel;
}

export interface Skill {
  id: number;
  name: string;
  description?: string;
  skill_type: string;
  category: string;
  skill_version?: string;
  skill_source?: string;
  git_repo_url?: string;
  git_commit_sha?: string;
  filesystem_path?: string;
  tags: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_sync_at?: string;
  files?: SkillFile[];
}

export interface SkillContentResponse {
  skill_id: number;
  skill_name: string;
  load_level: SkillLoadLevel;
  content: string;
  estimated_tokens: number;
  files_available: string[];
}

export interface SkillRecommendation {
  skill_id: number;
  skill_name: string;
  confidence: number;
  reasoning: string;
}

export interface GitImportRequest {
  git_url: string;
  source_name?: string;
  branch: string;
  auto_update: boolean;
  update_frequency_hours?: number;
}

export interface GitImportResponse {
  message: string;
  source_id: number;
  source_name: string;
  skills_discovered: number;
  skills: Array<{ id: number; name: string }>;
  commit_sha?: string;
}
