# Tool-Surface Deep Review — why "Good Morning" costs ~11,000 tokens

**Date:** 2026-07-23 · **Trigger:** PRD-207 voice latency (5–12s/turn) traced to tool loading; Gerard: "Why does 40 tools load for 'Hey Good Morning'? … Must be a way WHEN REQUIRED Auto can see all Platform tools… I assumed that's what the tool registry was for."
**Method:** 3 parallel static readers over the full selection stack (inventory / wiring / capability-safety). No code changed. All anchors verified against live source.

---

## 1. The bill — what one typical admin text turn actually ships

| Component | Count | ~Tokens | Query-gated? |
|---|---|---|---|
| **Promoted first-class action schemas** | **44** | **~8,710** | **NO — never** |
| Core ToolRegistry tools (general-context survivors) | ~11 | ~1,000–1,175 | No (context-category only) |
| `platform_execute` dispatcher (enum top-15) | 1 | ~280 | Yes (semantic top-K) |
| skill_tools | 0 | 0 | — (dead: always `[]`, `service.py:783`) |
| **Tool block total** | **~56** | **~10,000–10,200** | |
| + `PlatformActionsSection` prompt-text catalog | — | ~700–900 (narrowed) / **~4,000 (any fallback)** | Yes, but separate rank call |
| + Composio (text chat, apps connected) | varies | up to tens of k (historic 24–36k) | SDK-side search |

- **85% of the tool block is the 44 promoted schemas, and they are the one component with zero relevance gating** — `to_first_class_schemas()` filters by admin tier only (`tool_router.py:617-621`).
- The semantic narrowing everyone assumed covers the surface (PRD-138) touches **only the dispatcher enum**: ~1,100 → ~280 tokens. It narrows the cheap part and never the expensive part.
- Voice turns ride the same path (minus Composio via `VOICE_LIVE_SKIP_COMPOSIO`) — this block is roughly half the measured 5–12s voice first-frame; the rest is the model working through the bloat.

### Promoted grew into a second, ungated surface
46 actions carry `promoted=True` (44 visible to admin; 2 super-only): codegraph 9, workspace 8, blog 6, marketplace 5, agents 4, graph 3, workspace-misc 3, autonomy 2, field 2, shopify 2, analytics 1, search 1. Full table in §A1.

**History:** `docs/reviews/COMPOSIO-TOOL-REGRESSION-REVIEW.md:174-194,395-410` already identified PRD-122 (the change that created first-class promoted schemas, ~18→~33 tools) as a contributor to the "agents went dumb" regression. Promotion has since grown to 46. **This review reverses a documented regression, not a considered design.**

---

## 2. Failure posture — every fallback fails OPEN wider

Confirmed at every seam: when selection *can't* decide, the system ships the **maximum** surface:

| Bypass | Anchor | Resulting surface |
|---|---|---|
| Promoted first-class (by design) | `tool_router.py:614-624`, dispatcher built `exclude_promoted=True` `:572` | 44 schemas every turn, query-independent |
| **Query-embed timeout >2.5s** | `action_semantic_index.py:155-170,221-228` → `rank_actions` returns `[]` → `allowed_names=None` | **Full 132/137-action enum + full ~4k-token catalog** (seen repeatedly in prod logs during live calls) |
| `SEMANTIC_TOOL_ROUTING=false` / empty query / rank error / empty intersection | `tool_router.py:208-217`, `action_registry.py:205-225` | Full enum + full catalog |
| Heartbeat & task-execution `build_context` calls pass **no query** | `agent_factory.py:924-929`, `heartbeat_service.py:710-714` | Full 137 enum on **every** heartbeat/task run, forever |
| Embedding provider unset → `DeterministicEmbeddingProvider` (hash vectors) | `embedding_manager.py:66-144` | Ranking silently degrades (previously measured 97.9%→57.4% in-set); fail-open masks it |

A slow embed — the moment the system is already latency-stressed — is precisely when it ships the *most* tokens. Backwards on both axes (cost and quality).

---

## 3. Bugs found on the way (real, independent of any redesign)

1. **The full chat path discards the tools it computes.** `service.py:2120` builds `all_tools` with `is_super_admin` + PRD-221 `page_actions` threaded in → passed to `smart_chat.prepare(available_tools=…)` which **ignores it** ("kept for backward compatibility — ContextService loads tools internally", `smart_orchestrator.py:175-176`). The surface the LLM actually gets is rebuilt in `ToolsSection._load_filtered` (`tools.py:186-191`) **without `page_actions` and without `is_super_admin`**. ⇒ **PRD-221 S4 page-prior never reaches the main lane; su widening doesn't either.**
2. **Double ranking per turn** — `PlatformActionsSection` and the enum path each call `rank_actions` for the same query (`platform_actions.py:247` vs `tool_router.py:230`). Warm-cache second call, but two seams doing one job = drift risk (and they already disagree on fallback size).
3. **skill_tools is dead plumbing** — `_load_agent_context` hardcodes `[]` (`service.py:783`).
4. **`intent_classifier` hints name promoted actions** (`platform_list_agents`) that the hint-matcher can never match (promoted aren't in `available_tools` names it filters) — silent no-op (`intent_classifier.py:417-419` → `smart_tool_router.py:174-198`).
5. **ATOM vs full inconsistency** — ATOM turns already ship **zero** promoted schemas (dispatcher only, `service.py:2149-2156`). So Auto's capability surface silently flips with the complexity classifier: same user, same question phrasing differences → different tool world.
6. **No in-flight dedup on query embeds** — concurrent same-query turns each launch an embed (`action_semantic_index.py:134-177`).

---

## 4. Capability-safety verdicts (what a redesign can and cannot break)

**PROVEN SAFE — execution never depended on the tool list.**
- `unified_executor.py:713` resolves actions via unfiltered `registry.get()`; required-param validation `:724-742`; PRD-143 super-admin gate (`platform_executor.py:682-698`), admin gate `:706-739`, hierarchy `:831-902`, rate limits `:906-929`, destructive backstop `:931-948` — **all at execution time**, independent of surfacing.
- Direct `platform_*` calls route and gate even when the schema was never in the tool list (`unified_executor.py:758-765`). Absence from the surface degrades *discovery only*, never enforcement or executability.

**TRAP CONFIRMED — the dispatcher enum ceiling.**
- `to_dispatcher_schema` drops promoted names from `valid_actions` **before** the allow-list intersects (`action_registry.py:196-225`) — `allowed_names` can never re-admit a promoted action, and the empty-intersection fallback is also promoted-excluded. A test pins this (`test_action_registry_filtered.py:329`).
- ⇒ Naively deleting the first-class loading leaves 46 actions **invisible + unlisted + undescribed** on every surface. The fix must flip `exclude_promoted` at the enum/catalog build sites in the same change.

**STRUCTURAL DEPENDENCY — the 8 `workspace_*` file tools.**
- Their ToolRegistry fallbacks are deactivated ("SUPERSEDED by workspace_* promoted actions", `tool_registry.py:760-764`). Coder/mission templates instruct agents to call them by name (`templates.py:794-841,1256-1293`). They must stay guaranteed-discoverable in mission/coder contexts.

**SOFT prose dependencies** — cadence prompts (`auto_cadence.py:50,60`), onboarding seeds (`seed_onboarding_agents.py:91-217`), mission template prose name promoted actions; they degrade gracefully **iff** those names are present in the dispatcher enum (or re-surfaced on demand).

**Test pins** — ~10 files assert today's surface semantics (heaviest: `test_action_registry_filtered.py`, `test_tool_router_semantic.py`, `test_action_semantic_index.py` exact-count assertions, `test_prd143_su_*`). Full impact table in §A2.

---

## 5. Target design — "the registry becomes what Gerard assumed it was"

> Requirement (Gerard): Auto has access to **all** tools; never loads 40+ schemas a turn; **when required** Auto can see everything available, with descriptions, and pick the best itself.

### Tier 0 — small, stable, always-on (cache-friendly)
1. ~11 core ToolRegistry tools (unchanged).
2. **`platform_execute` dispatcher** — enum = semantic top-K **with a relevance floor** (absolute min + relative-to-best ratio; greeting ⇒ 0–3 names), **promoted names now eligible** (flip `exclude_promoted` once first-class loading is gated), unioned with: page-manifest actions (PRD-221 — fixed to actually arrive), context pins (the 8 `workspace_*` in mission/coder contexts), and a handful of telemetry-earned hot actions.
3. **NEW `platform_find_tools(query)`** — the "WHEN REQUIRED" seam. Searches the action registry (`rank_actions` with `exclude_promoted=False`, generous top-N) and returns matches: name, description, full param schema, permission notes, examples. Auto reads → calls `platform_execute(action, params)`. Zero new tables; standard 3-file action registration; reuses the existing index. *This makes the full catalog reachable in one hop from any turn without ever shipping it wholesale.*

### Tier 1 — earned first-class (optional hybrid)
Actions ranking above a high-confidence bar for **this turn's** query (cap ~6) also get real first-class schemas — common flows keep one-hop calling with typed params. Everything else lives behind dispatcher + find_tools.

### Posture flip — fail CLOSED-small
On embed timeout / rank failure / no query: surface = core tools + dispatcher with **pin-set enum** (memory, resume_context, page actions, find_tools) + find_tools — **not** 137 names + 4k catalog. The catalog section on fallback renders one line ("discover more via platform_find_tools"), not 4k of text.

### Consolidations riding along
- **One rank call per turn**, shared by enum + catalog + (new) first-class gate.
- Fix dead plumbing: thread `page_actions`/`is_super_admin` into the surface that actually ships (ToolsSection), or make the full path consume `_get_tools` output — one seam, not two.
- Thread `task_description` as query on heartbeat/task-execution `build_context` calls.
- ATOM and full paths converge on the same Tier-0 surface (ends the classifier-flips-capability inconsistency).

### Expected effect (per turn, typical)
| | Today | Target |
|---|---|---|
| Tool block | ~10,000–10,200 tok | **~1,500–2,500 tok** |
| Catalog text | 700–900 / 4,000 fallback | ~300 / ~50 fallback |
| Fallback posture | fail-open to max | fail-closed to pins |
| Voice first-frame | 5–12s | expect ~2–4s (tool-load share ≈ gone; same path as text) |
| Selection quality | 44 constant distractors | distractor-free + on-demand catalog |

---

## 6. Staged rollout (each stage independently shippable + revertible)

**PR-A — bug fixes, no design change:** dead-plumbing fix (page_actions/su threading), heartbeat/task query threading, single-rank consolidation, embed in-flight dedup. CI: update nothing conceptually — behavior becomes what PRD-221/PRD-143 already claimed.

**PR-B — additive, flags default-off:** `platform_find_tools` action; relevance floor (`SEMANTIC_TOOL_ROUTING_FLOOR`, `…_FLOOR_RATIO`); fail-closed-small fallback (`TOOL_FALLBACK_MODE=open-full|closed-pins`, default `open-full`); shadow telemetry — log the would-be surface per turn (`ToolSignalRecorder` extension) without changing what ships.

**PR-C — the flip:** gate promoted first-class by relevance (`TOOL_SURFACE_PROMOTED_GATED=true`), enum/catalog stop excluding promoted, fallback flips to closed-pins. Test updates land here (~10 files, §A2). Revert = one flag.

**Eval gate between B and C (no local runs — CI + shadow telemetry in prod):** replay shadow logs over real traffic for a day or two: (1) tool-intent turns — chosen action would still have been surfaced ≥ baseline 97.9% in-set rate; (2) greeting-class turns — surface ≤ 3 actions; (3) zero occurrences of a turn that *executed* an action the gated surface wouldn't have listed (executor logs make this measurable).

---

## 7. Open decisions (Gerard)

1. **Hybrid first-class** (top-ranked ≤6 get typed schemas) vs dispatcher-only + find_tools? *Rec: hybrid — keeps hot paths one-hop.*
2. **Fallback posture** after eval: default `closed-pins`? *Rec: yes — slow embed should never mean max tokens.*
3. **`workspace_*` pinning:** always-on for mission/coder agent types via existing category machinery? *Rec: yes (structural dependency, §4).*
4. **Catalog text:** keep a tiny top-8 list + find_tools pointer, or drop the section entirely? *Rec: keep tiny — costs ~300 tok, preserves zero-shot discoverability.*
5. **Promotion diet:** demote never-used promoted actions outright (blog×6? shopify×2?) using `tool_execution_logs` usage — data pull first? *Rec: separate, after telemetry.*

---

## A1. Promoted inventory (46)

platform_list_agents, platform_get_agent, platform_create_agent, platform_update_agent (agents) · platform_get_activity_feed (analytics) · platform_get_autonomy_level, platform_set_autonomy_level^su (autonomy) · platform_list_blog_posts, platform_get_blog_post, platform_create_blog_post, platform_update_blog_post, platform_publish_blog_post, platform_generate_cover_image (blog) · platform_codegraph_list_projects, _get_symbol, _search, _call_graph, _dependencies, _architecture, _index, _reindex, _set_auto_reindex (codegraph) · platform_field_inject, platform_field_query (field) · platform_query_graph, platform_graph_neighbors, platform_graph_path (graph) · platform_browse_marketplace_agents, _plugins, _skills, platform_install_plugin, platform_install_skill (marketplace) · platform_search_memory (search) · platform_shopify_sync_catalog, platform_shopify_sync_status (shopify) · platform_get_system_health^su, platform_resume_context, platform_store_memory (workspace-misc) · workspace_read_file, workspace_write_file, workspace_list_dir, workspace_grep, workspace_exec, workspace_git, workspace_get_public_url, workspace_html_to_png (workspace).
Size: ~35.8k chars ≈ **8.9k tokens** (all 46); admin-visible 44 ≈ **8.7k tokens**.

## A2. Test-impact map

| File | Pins | Touched by |
|---|---|---|
| test_action_registry_filtered.py:244-350 | enum semantics incl. promoted-excluded-even-in-allowlist (:329) | flip + floor |
| test_tool_router_semantic.py:427-720 | narrowed enum contents; rank-raise→full-enum invariant (:563) | floor + posture |
| test_action_semantic_index.py:214-408 | exact result counts (:249,:261); timeout→[] (:366) | floor |
| test_prd143_su_surface.py:322-338 | su never leaks; `assert names` non-empty (:329) | floor (empty-result case) |
| test_prd143_selection_at_scale.py:391-460 | in-set metric across intents | floor + flip |
| test_us015_registry_intent_filter.py / test_us014_graph_router_delegation.py | promoted-pin union behavior | flip |
| test_prd143_su_registry.py:80-190 | to_first_class_schemas su/admin filters | flip |
| test_prd143_concierge_journey.py:256-257 | surface = first_class + dispatcher composition | flip |

## A3. Evidence trail
Three reader reports (inventory / wiring / safety) captured in session transcripts 2026-07-23; all file:line anchors re-verifiable by grep. Key prod-log evidence: `voice_live_ws_first_frame ms=12292/13398`, "narrowing falls back to the full enum" warnings during live calls, "Available tools: [37 names]" per turn, `updates=469` noise-turn storm (pre-STT-retune).
