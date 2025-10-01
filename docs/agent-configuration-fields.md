# Agent Configuration Fields

## Database vs UI Fields Mapping

After reviewing the agent configuration implementation, I've identified a mismatch between what the UI is displaying and what the database is storing. This document outlines the discrepancies and provides recommendations for handling them.

### Fields in Database Model

The database has the following fields for agents:

- `id` (Integer, primary key)
- `name` (String)
- `description` (Text)
- `agent_type` (String) - 'custom', 'system', 'specialized'
- `status` (String) - 'active', 'inactive', 'training'
- `configuration` (JSON) - A flexible JSON field for agent-specific config
- `performance_metrics` (JSON) - Performance data

Fields added specifically for UI requirements:
- `priority_level` (String) - 'low', 'medium', 'high', 'critical'
- `max_concurrent_tasks` (Integer) - default: 5
- `auto_start` (Boolean) - default: false

Timestamp fields:
- `created_at` (DateTime)
- `updated_at` (DateTime)
- `created_by` (String)

### Fields in UI Configuration

The UI is displaying and allowing configuration of the following fields:

#### Basic Configuration
- `name` - Agent name
- `description` - Agent description
- `priority_level` - Priority level
- `auto_start` - Auto start on system boot

#### Performance Settings
- `max_concurrent_tasks` - Maximum number of concurrent tasks
- `task_timeout` - Task timeout in seconds
- `retry_attempts` - Number of retry attempts
- `enable_caching` - Enable caching for performance

#### Security & Access
- `access_level` - Access level
- `enable_logging` - Enable logging
- `enable_rate_limiting` - Apply rate limits
- `api_rate_limit` - API calls per minute

#### Memory & Storage
- `memory_limit` - Memory limit in MB
- `context_window_size` - Context window size in tokens
- `persistent_memory` - Retain memory between sessions
- `memory_cleanup_interval` - Cleanup interval in hours

#### Advanced Configuration
- `custom_config` - Custom JSON configuration

### Mismatch Analysis

1. **Direct Database Fields**:
   - `name`, `description`, `priority_level`, `max_concurrent_tasks`, `auto_start` have direct mappings to database fields

2. **Fields Expected to be in JSON Configuration**:
   - All Performance Settings (except `max_concurrent_tasks`)
   - All Security & Access settings
   - All Memory & Storage settings
   - Advanced Configuration

3. **Missing Database Support**:
   The database doesn't explicitly define the structure for many UI settings, relying instead on the JSON `configuration` field for flexible storage.

## Recommendations

1. **Short-term Solution**: Continue using the JSON `configuration` field in the database to store all the additional UI settings. The current API seems to support this approach through the `useUpdateAgentConfig` hook.

2. **Medium-term Solution**: Document the expected structure of the JSON `configuration` field to ensure consistency between the UI and API.

3. **Long-term Solution**: Consider one of the following approaches:
   - Add explicit columns to the database for frequently used configuration fields
   - Create a separate `agent_configuration` table with a one-to-one relationship to the `agents` table
   - Define a clear schema for the JSON `configuration` field and enforce it at the API level

## Implementation Status

Currently, the UI is configured to handle all these fields, but we need to ensure:
1. The API properly saves all fields to the JSON `configuration` field
2. The API correctly retrieves and returns these fields when requested
3. The UI properly displays the returned values

This document should be updated as the implementation evolves.
