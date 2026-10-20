/**
 * PRD-261 A-19a — the look scope is the host's look runtime's (lib/look-kit/scope.ts): the host's
 * AccountLookScope, its look scope context and builtin Play's look storage import it there. This
 * path re-exports it for Play's own code (lib/play/seen.ts) until builtin Play leaves core.
 */
export {
  DEFAULT_LOOK_SCOPE,
  type KeyMove,
  lookScope,
  moveUnscopedOnce,
  NO_USER,
  NO_WORKSPACE,
  SCOPE_SEPARATOR,
  scopedKey,
} from '../look-kit/scope'
