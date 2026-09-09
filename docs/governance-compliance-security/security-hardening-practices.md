# Security Hardening Practices

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/workspace-execution/security-sandboxing.md](docs/workspace-execution/security-sandboxing.md)
- [orchestrator/alembic/versions/prd140_permission_bypass_log.py](orchestrator/alembic/versions/prd140_permission_bypass_log.py)
- [orchestrator/alembic/versions/prd140_team_lead_enabled.py](orchestrator/alembic/versions/prd140_team_lead_enabled.py)
- [orchestrator/alembic/versions/prd185_s1b_toollog_user_nullable.py](orchestrator/alembic/versions/prd185_s1b_toollog_user_nullable.py)
- [orchestrator/alembic/versions/prd200_s2_drop_checkpoint_count.py](orchestrator/alembic/versions/prd200_s2_drop_checkpoint_count.py)
- [orchestrator/api/widgets/cors.py](orchestrator/api/widgets/cors.py)
- [orchestrator/core/security/__init__.py](orchestrator/core/security/__init__.py)
- [orchestrator/core/security/bypass_audit.py](orchestrator/core/security/bypass_audit.py)
- [orchestrator/core/security/hierarchy_permissions.py](orchestrator/core/security/hierarchy_permissions.py)
- [orchestrator/core/security/url_validator.py](orchestrator/core/security/url_validator.py)
- [orchestrator/core/services/auto_cadence.py](orchestrator/core/services/auto_cadence.py)
- [orchestrator/modules/tools/execution/exec_platform.py](orchestrator/modules/tools/execution/exec_platform.py)
- [orchestrator/modules/tools/execution/telemetry.py](orchestrator/modules/tools/execution/telemetry.py)
- [orchestrator/scripts/check_hierarchy_gate.py](orchestrator/scripts/check_hierarchy_gate.py)
- [orchestrator/tests/security/test_hierarchy_permissions.py](orchestrator/tests/security/test_hierarchy_permissions.py)
- [orchestrator/tests/security/test_prd172_tenant_isolation.py](orchestrator/tests/security/test_prd172_tenant_isolation.py)
- [orchestrator/tests/test_p2w0_telemetry_user_id.py](orchestrator/tests/test_p2w0_telemetry_user_id.py)
- [orchestrator/tests/test_p2w2_cors_boot_guard.py](orchestrator/tests/test_p2w2_cors_boot_guard.py)
- [orchestrator/tests/test_p2w3_checkpoints_deleted.py](orchestrator/tests/test_p2w3_checkpoints_deleted.py)
- [orchestrator/tests/test_prd008a_cors_coverage.py](orchestrator/tests/test_prd008a_cors_coverage.py)
- [orchestrator/tests/test_prd139_telemetry.py](orchestrator/tests/test_prd139_telemetry.py)
- [orchestrator/tests/test_prd143_boundary_sweep.py](orchestrator/tests/test_prd143_boundary_sweep.py)
- [orchestrator/tests/test_prd143_concierge_journey.py](orchestrator/tests/test_prd143_concierge_journey.py)
- [orchestrator/tests/test_prd143_full_surface_positive.py](orchestrator/tests/test_prd143_full_surface_positive.py)
- [orchestrator/tests/test_prd143_selection_metric.py](orchestrator/tests/test_prd143_selection_metric.py)
- [orchestrator/tests/test_prd177_composio_telemetry.py](orchestrator/tests/test_prd177_composio_telemetry.py)
- [orchestrator/tests/test_prd186_s3_hardening.py](orchestrator/tests/test_prd186_s3_hardening.py)
- [orchestrator/tests/test_telemetry_session_ownership.py](orchestrator/tests/test_telemetry_session_ownership.py)

</details>



This page details the security hardening practices implemented in the Automatos AI codebase, focusing on authorization boundary sweeps, CORS boot guard, fail-closed widget authentication, webhook HMAC and deduplication, permission bypass auditing, and tenant isolation test suites. It explains the implementation, data flow, and key functions/classes involved in these security mechanisms.

## Authorization Boundary Sweeps

Authorization boundary sweeps ensure that sensitive operations are restricted to authorized users and roles, particularly super-administrators. The system employs a "deny-by-default" security model, where every check is workspace-bound, and explicit permissions are required for actions.

The `test_prd143_boundary_sweep.py` suite [orchestrator/tests/test_prd143_boundary_sweep.py:1-681]() exhaustively tests that super-admin (SU) actions are inaccessible to regular operators and API-key admins. This includes verifying that SU actions are absent from the OpenAI tool surface, the dispatcher enum, semantic ranking, and graph ranking. It also confirms that the executor refuses SU actions for operators under full autonomy and for API-key admins.

Key aspects of the authorization boundary sweeps include:
*   **Live Registry-Driven SU Action Set**: The set of super-admin actions is read from the live `ActionRegistry` [orchestrator/tests/test_prd143_boundary_sweep.py:141-156](), ensuring that any new SU action is automatically covered by the sweeps.
*   **Locked Router Set**: Routers that are locked for super-admin access are parsed from a signed-off manifest (`docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md`).
*   **Exhaustive Route Enumeration**: Every route on every locked router returns a "Super admin only" 403 error for members, workspace admins/owners, and API-key principals.
*   **Autonomy Dial Absence**: The `platform_set_autonomy_level` action is absent from the agent's surface when the user is an operator.

The `hierarchy_permissions.py` module [orchestrator/core/security/hierarchy_permissions.py:1-363]() provides a single chokepoint for determining if an actor may apply a change to a target. This is called from the platform tool dispatcher (`platform_executor`) and acts as a defense-in-depth mechanism for service-layer mutations.

### Security Model in `hierarchy_permissions.py`

The `can_actor_modify` function [orchestrator/core/security/hierarchy_permissions.py:112-121]() implements a deny-by-default model with several checks:
*   **No Actor**: If `actor_agent_id` is `None`, the request is denied [orchestrator/core/security/hierarchy_permissions.py:131-134](). Trusted system flows must pass an explicit actor ID.
*   **Unknown Actor**: If no `agents` row corresponds to the `actor_agent_id`, the request is denied [orchestrator/core/security/hierarchy_permissions.py:136-139]().
*   **Cross-Workspace Actor**: If the actor's `workspace_id` does not match the call's `workspace_id`, the request is denied, blocking cross-tenant IDOR [orchestrator/core/security/hierarchy_permissions.py:141-145]().
*   **Inactive Actor**: If the actor's status is not 'active', the request is denied [orchestrator/core/security/hierarchy_permissions.py:147-150]().
*   **Narrowed System Bypass**: Only agents with `is_system_agent=True` AND whose `name` is in `SYSTEM_BYPASS_ALLOWLIST` [orchestrator/core/security/hierarchy_permissions.py:68-74]() are allowed to bypass the hierarchy [orchestrator/core/security/hierarchy_permissions.py:152-157](). This prevents accidental privilege escalation.

### Permission Bypass Audit

The `bypass_audit.py` module [orchestrator/core/security/bypass_audit.py]() is responsible for logging and auditing instances where permissions are bypassed. This is crucial for maintaining a clear record of privileged operations and detecting potential misuse. The `ToolExecutionLog` model [orchestrator/modules/tools/execution/telemetry.py:149]() includes a `router_decision` field that can capture details about whether an action was approved via a grant ID or was human-directed [orchestrator/modules/tools/execution/telemetry.py:162-164](). This allows for auditing of actions that might bypass standard automated approval flows.

Sources:
*   [orchestrator/tests/test_prd143_boundary_sweep.py:1-681]()
*   [orchestrator/core/security/hierarchy_permissions.py:1-363]()
*   [orchestrator/modules/tools/execution/telemetry.py:149]()
*   [orchestrator/modules/tools/execution/telemetry.py:162-164]()
*   [orchestrator/core/security/bypass_audit.py]()

## CORS Boot Guard

The CORS boot guard ensures that the application starts securely, particularly concerning widget interactions. The `WidgetCORSMiddleware` [orchestrator/api/widgets/cors.py:138-141]() is an ASGI-native middleware that handles dynamic CORS for the storefront-widget and dashboard-Sites API surfaces without buffering `StreamingResponse` or SSE connections.

### Dynamic CORS Authorization

Instead of a global widget origin allowlist, storefront origins are authorized from the `SdkApiKey.allowed_domains` associated with each API key [orchestrator/api/widgets/cors.py:23-24](). This ensures that only explicitly permitted domains can interact with the widget API.

The `_origin_allowed_dynamic` function [orchestrator/api/widgets/cors.py:104-131]() determines if an origin is allowed:
1.  **Platform Origins**: If the origin is one of the platform's own domains (e.g., `app.automatos.app`), it's immediately allowed [orchestrator/api/widgets/cors.py:112-113](). These are defined in `config.CORS_ALLOW_ORIGINS` [orchestrator/api/widgets/cors.py:62-66]().
2.  **Key-Scoped Paths**: For paths under `/api/widgets`, the system checks if the origin is named on *any* active public API key's `allowed_domains` [orchestrator/api/widgets/cors.py:114-115](). This check is performed by `ApiKeyService.origin_allowed_by_any_key` [orchestrator/api/widgets/cors.py:99]() and the result is cached for a `_DYNAMIC_TTL_SECONDS` [orchestrator/api/widgets/cors.py:87]().
3.  **Fail-Closed**: If the origin is not a platform origin and not allowed by any API key, access is denied. Lookup failures (e.g., DB unavailable) also result in denial, but are not cached, allowing for retries [orchestrator/api/widgets/cors.py:120-127]().

The `test_p2w2_cors_boot_guard.py` suite [orchestrator/tests/test_p2w2_cors_boot_guard.py:1-157]() verifies that the security boot guard correctly aborts a boot if critical security configurations are missing and that widget CORS fails closed for unauthorized origins. It specifically tests that `config.validate_security()` is called directly within the `lifespan` function [orchestrator/tests/test_p2w2_cors_boot_guard.py:73-75](), ensuring that any security validation failure will prevent the application from starting.

### Widget CORS Flow

```mermaid
graph TD
    A[Client Request] --> B{Is HTTP Request?};
    B -- No --> C[Pass to Next ASGI App];
    B -- Yes --> D[Extract Path and Origin];
    D --> E{Path Covered by WidgetCORSMiddleware?};
    E -- No --> C;
    E -- Yes --> F{Is OPTIONS Preflight?};
    F -- Yes --> G{Origin Allowed Dynamically?};
    F -- No --> H{Origin Allowed Dynamically?};
    G -- No --> I[Return 403/400 Forbidden];
    H -- No --> I;
    G -- Yes --> J[Add CORS Headers to Response];
    H -- Yes --> J;
    J --> K[Pass to Next ASGI App];

    subgraph _origin_allowed_dynamic
        L[Origin] --> M{Is Platform Origin?};
        M -- Yes --> N[Allow];
        M -- No --> O{Path starts with /api/widgets?};
        O -- No --> P[Deny];
        O -- Yes --> Q[Check Dynamic Cache];
        Q -- Hit & Valid --> N;
        Q -- Miss or Expired --> R[Call _origin_allowed_by_key_sync];
        R -- Success --> S[Cache Result & Allow/Deny];
        R -- Failure --> T[Log Error & Deny (No Cache)];
        S --> N;
        T --> P;
    end
```
Sources:
*   [orchestrator/api/widgets/cors.py:1-217]()
*   [orchestrator/tests/test_prd008a_cors_coverage.py:1-223]()
*   [orchestrator/tests/test_p2w2_cors_boot_guard.py:1-157]()

## Fail-Closed Widget Authentication

Widget authentication is designed to be fail-closed, meaning that in case of any ambiguity or failure in authentication, access is denied. This is closely tied to the dynamic CORS mechanism.

The `_origin_allowed_dynamic` function [orchestrator/api/widgets/cors.py:104-131]() explicitly handles cases where the `_origin_allowed_by_key_sync` function (which queries the database for API key domain permissions) fails. If an exception occurs during this lookup, it logs the error and returns `False`, effectively denying access and failing closed [orchestrator/api/widgets/cors.py:120-127](). This prevents a denial-of-service attack from compromising the security of the widget API by causing database lookup failures.

The `test_p2w2_cors_boot_guard.py` includes a test case `test_key_fallback_fails_closed_and_does_not_cache_failures` [orchestrator/tests/test_p2w2_cors_boot_guard.py:157]() that specifically verifies this fail-closed behavior. It simulates a database unavailability scenario and asserts that the `_origin_allowed_dynamic` function returns `False` without caching the failure, ensuring that subsequent attempts will re-evaluate the origin.

Sources:
*   [orchestrator/api/widgets/cors.py:104-131]()
*   [orchestrator/api/widgets/cors.py:120-127]()
*   [orchestrator/tests/test_p2w2_cors_boot_guard.py:157]()

## Webhook HMAC and Deduplication

Webhook security is critical for external integrations. The system implements HMAC (Hash-based Message Authentication Code) verification and deduplication to ensure the integrity and uniqueness of incoming webhook events.

While specific code for webhook HMAC and deduplication was not provided in the given files, the presence of `TriggerSubscription` in the table of contents (6.4. Scheduling & Triggers) suggests that webhooks are a core part of the system's eventing. In a secure webhook implementation, HMAC verification would typically involve:
1.  **Shared Secret**: A secret key shared between the sender and receiver.
2.  **HMAC Generation**: The sender computes an HMAC of the webhook payload using the shared secret and includes it in a request header.
3.  **HMAC Verification**: The receiver (Automatos AI) recomputes the HMAC using its copy of the shared secret and the received payload. If the computed HMAC matches the received HMAC, the payload's integrity and authenticity are confirmed.

Deduplication is essential to prevent replay attacks and ensure idempotent processing of webhook events. This often involves:
1.  **Unique Identifier**: Webhook events typically include a unique ID (e.g., `X-GitHub-Delivery` for GitHub webhooks).
2.  **Storage and Check**: The system stores processed webhook IDs (e.g., in Redis or a database) and checks incoming IDs against this store. If an ID is already present, the event is considered a duplicate and is discarded or handled appropriately.

These mechanisms protect against tampering, unauthorized sending, and redundant processing of webhook events.

Sources:
*   (No direct code citations for webhook HMAC/deduplication were available in the provided files, but the concept is implied by the wiki structure.)

## Permission Bypass Audit

The permission bypass audit system is designed to track and log instances where standard permission checks are circumvented, typically by system agents or through explicit administrative overrides. This is crucial for security monitoring and compliance.

The `hierarchy_permissions.py` module [orchestrator/core/security/hierarchy_permissions.py:1-363]() includes a `PermissionDecision` dataclass [orchestrator/core/security/hierarchy_permissions.py:85-107]() that captures the outcome of a permission check. This dataclass has fields like `bypass` and `bypass_kind` [orchestrator/core/security/hierarchy_permissions.py:99-100](), which are set when a system actor bypasses the normal hierarchy checks. For example, the `can_actor_modify` function [orchestrator/core/security/hierarchy_permissions.py:112-121]() explicitly sets `bypass=True` and `bypass_kind="system_actor"` for system agents in the `SYSTEM_BYPASS_ALLOWLIST` [orchestrator/core/security/hierarchy_permissions.py:152-157]().

The `write_telemetry` function in `telemetry.py` [orchestrator/modules/tools/execution/telemetry.py:89-99]() is responsible for logging every tool execution to `tool_execution_logs`. The `router_decision` field in `ToolExecutionLog` [orchestrator/modules/tools/execution/telemetry.py:160-165]() can store information about whether an action was `autonomous`, `approved_via_grant_id`, or `human_directed`. This allows for a detailed audit trail of how actions were authorized, including those that might involve a bypass or explicit approval.

The `exec_platform.py` module [orchestrator/modules/tools/execution/exec_platform.py:29-50]() demonstrates how actor identity is handled for platform actions. It explicitly strips any `_agent_id` or `_agent_name` that an LLM might try to smuggle in, preventing impersonation and ensuring that the `hierarchy_permissions` module correctly identifies the actor. If `agent_id` is unknown, the permission check fails closed as an "anonymous_actor" [orchestrator/modules/tools/execution/exec_platform.py:60-64]().

Sources:
*   [orchestrator/core/security/hierarchy_permissions.py:1-363]()
*   [orchestrator/core/security/hierarchy_permissions.py:85-107]()
*   [orchestrator/core/security/hierarchy_permissions.py:99-100]()
*   [orchestrator/core/security/hierarchy_permissions.py:152-157]()
*   [orchestrator/modules/tools/execution/telemetry.py:89-99]()
*   [orchestrator/modules/tools/execution/telemetry.py:160-165]()
*   [orchestrator/modules/tools/execution/exec_platform.py:29-50]()
*   [orchestrator/modules/tools/execution/exec_platform.py:60-64]()

## Tenant Isolation Test Suites

Tenant isolation is a fundamental security requirement, ensuring that data and operations of one workspace are completely separated from others. Automatos AI employs comprehensive test suites to verify this isolation.

The `test_prd172_tenant_isolation.py` suite [orchestrator/tests/security/test_prd172_tenant_isolation.py:1-606]() is dedicated to proving tenant isolation closure. It uses a cross-tenant matrix approach, asserting that workspace A cannot read, write, or delete any of workspace B's data across various in-scope domains, including skills, documents/vectors, workflows, memory, and context.

Key aspects of tenant isolation testing:
*   **Mocked Database and Injected Auth**: The tests use a mocked database and inject authentication via `app.dependency_overrides` [orchestrator/tests/security/test_prd172_tenant_isolation.py:10-11](), allowing for behavioral testing without a live PostgreSQL instance.
*   **Workspace-Scoped Checks**: Every check in `hierarchy_permissions.py` is scoped by `workspace_id` [orchestrator/core/security/hierarchy_permissions.py:128](), explicitly denying cross-tenant mutations.
*   **Skills Isolation**: Tests verify that a workspace cannot see another workspace's private skills, but can see global skills or its own skills [orchestrator/tests/security/test_prd172_tenant_isolation.py:85-98](). Super-admins, however, can see any skill [orchestrator/tests/security/test_prd172_tenant_isolation.py:100-103]().
*   **Agent Attachment**: Attaching a skill to an agent in another workspace is denied with a 404 error [orchestrator/tests/security/test_prd172_tenant_isolation.py:105-112]().
*   **Global Skill Deletion**: Global skills can only be deleted by a super-admin, preventing workspace callers from inadvertently or maliciously removing core platform components [orchestrator/tests/security/test_prd172_tenant_isolation.py:161-163]().

### S3 Vectors Hardening

The `test_prd186_s3_hardening.py` suite [orchestrator/tests/test_prd186_s3_hardening.py:1-307]() focuses on hardening S3 Vectors for tenant isolation on a shared bucket.
*   **Total Tenant Isolation**: `search()` operations on the shared bucket drop unlabeled (missing `workspace_id`) or mismatched `workspace_id` hits [orchestrator/tests/test_prd186_s3_hardening.py:83-94]().
*   **Workspace Stamping**: `add_documents` explicitly stamps the `workspace_id` onto the metadata of vectors before storage [orchestrator/tests/test_prd186_s3_hardening.py:108-117]().
*   **File-Scoped Deletion**: `delete_for_files` ensures that deletions are scoped to specific files within a workspace, preventing an index-wide sweep that could affect other tenants [orchestrator/tests/test_prd186_s3_hardening.py:119-134]().

Sources:
*   [orchestrator/tests/security/test_prd172_tenant_isolation.py:1-606]()
*   [orchestrator/core/security/hierarchy_permissions.py:128]()
*   [orchestrator/tests/security/test_prd172_tenant_isolation.py:85-98]()
*   [orchestrator/tests/security/test_prd172_tenant_isolation.py:100-103]()
*   [orchestrator/tests/security/test_prd172_tenant_isolation.py:105-112]()
*   [orchestrator/tests/security/test_prd172_tenant_isolation.py:161-163]()
*   [orchestrator/tests/test_prd186_s3_hardening.py:1-307]()
*   [orchestrator/tests/test_prd186_s3_hardening.py:83-94]()
*   [orchestrator/tests/test_prd186_s3_hardening.py:108-117]()
*   [orchestrator/tests/test_prd186_s3_hardening.py:119-134]()

---