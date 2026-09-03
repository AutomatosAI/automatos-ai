/**
 * PRD-235 W2 — the folder a Code Canvas is rooted at.
 *
 * Roots are workspace-relative ('sessions/71', 'projects/my-repo'); '.' is the
 * workspace root. Anything absolute or escaping is refused — the worker would
 * refuse it too, this just keeps a bad deep link from producing a blank tree.
 */
export const WORKSPACE_ROOT = '.'

export function normalizeCodeRoot(input: string | null | undefined): string | null {
  if (input == null) return null
  let s = String(input).trim().replace(/\\/g, '/')
  if (s === '' || s === '.' || s === './') return WORKSPACE_ROOT
  if (s.startsWith('/')) return null
  s = s.replace(/^\.\//, '').replace(/\/+$/, '')
  if (s === '' ) return WORKSPACE_ROOT
  if (s.split('/').some((seg) => seg === '' || seg === '.' || seg === '..')) return null
  return s
}

export function canvasTitleFor(root: string): string {
  return root === WORKSPACE_ROOT ? 'Code Canvas' : `Code Canvas · ${root}`
}
