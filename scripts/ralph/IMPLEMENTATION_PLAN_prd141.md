# Implementation Plan — PRD-141 Widget Vertical-Agnostic Refactor

Source PRD: `docs/PRDS/141-WIDGET-VERTICAL-AGNOSTIC-REFACTOR.md`
Ralph PRD JSON: `scripts/ralph/prd.json` (staged from `prd-141-widget-vertical.json`)
Branch: `ralph/prd-141-widget-vertical-refactor`
Worktree: `/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-PRD-141`

Single source of truth for Ralph progress. Tick `- [x]` only after a story's acceptance criteria are fully satisfied AND the validation step from PROMPT_build_prd141.md passes.

## Phase 0 — Plugin scaffolding (zero-risk foundation)

- [x] US-001 — Scaffold plugin registry and base protocol
- [x] US-002 — Create generic pass-through plugin
- [x] US-003 — Create shim Shopify plugin that delegates to existing chat.py code

## Phase 1 — Lift Shopify into the plugin (the risky one)

- [x] US-004 — Capture proactive opener + cart-idle fixtures *(synthetic-but-realistic acceptable — see PROMPT_build_prd141.md "Special note for US-004"; human reviews after Ralph generates)*
- [x] US-011 — Add snapshot equivalence tests for product-page opener and cart-idle opener *(REORDERED ahead of the lifts — runs against the US-003 shim so byte-equality guards every move; must pass now and stay green through US-005–US-010)*
- [x] US-005 — Move PROACTIVE_OPENER_FIELDS constant into the Shopify integration folder
- [x] US-006 — Move _resolve_graph_related_products into Shopify plugin
- [x] US-007 — Move _resolve_cart_recommendations into Shopify plugin
- [x] US-008 — Move _build_proactive_opener_message and _build_cart_idle_opener_message into Shopify plugin
- [x] US-009 — Add Alembic migration backfilling workspace.settings.vertical for Shopify workspaces *(staging apply deferred to human; Ralph validates offline only)*
- [x] US-010 — Rewire chat.py to dispatch via PLUGIN_REGISTRY and delete inline Shopify functions
- [ ] US-012 — Add CI grep gate enforcing no Shopify keys in generic surfaces

## Phase 2 — SDK sends page_context on regular messages *(automatos-widget-sdk repo)*

- [ ] US-013 — Widen ChatRequest.page_context type to Record<string, unknown> *(CROSS-REPO — automatos-widget-sdk)*
- [ ] US-014 — Update SDK sendMessage to accept and forward pageContext *(CROSS-REPO — automatos-widget-sdk)*
- [ ] US-015 — Add SDK integration test asserting page_context flows *(CROSS-REPO — automatos-widget-sdk)*

## Phase 3 — Skill update + generic skill *(automatos-skills repo)*

- [ ] US-016 — Update shopify-support SKILL.md to v1.3.2 *(CROSS-REPO — automatos-skills)*
- [ ] US-017 — Create new generic-default-widget-support skill *(CROSS-REPO — automatos-skills)*

## Phase 4 — Docs

- [ ] US-018 — Write docs/integrations/README.md explaining how to add a new vertical
- [ ] US-019 — Write hypothetical barbershop walkthrough doc as architecture validation

## Release gate

- [ ] US-020 — Canary deploy Phase 1 on INBUILD workspace and verify 24h soak *(OPERATIONAL — human-only; Ralph must skip)*

## Notes for Ralph

- Phase 0 stories (US-001/002/003) are safe to run autonomously
- US-004 now executable with synthetic fixtures (see PROMPT_build_prd141.md "Special note for US-004"). Human reviews fixtures after this story before US-005 starts.
- US-011 (snapshot tests) is REORDERED ahead of US-005 so the byte-equality net exists before any lift. It runs against the US-003 shim and MUST pass before the lifts start; every lift story (US-005–US-010) must keep `cd orchestrator && python -m pytest integrations/` green or it broke equivalence.
- US-013/014/015 are in a different repo (automatos-widget-sdk); Ralph should mark BLOCKED-CROSS-REPO and exit (human will set up a separate Ralph worktree for SDK work)
- US-016/017 are in automatos-skills; same — BLOCKED-CROSS-REPO and exit
- US-020 is operational; Ralph should mark SKIPPED-HUMAN and exit
