/**
 * The free baseline skill library — the repo a fresh install imports first.
 *
 * A new install (local edition especially) ships with no skills in the
 * marketplace. Every `SKILL.md` in this repo imports as a marketplace skill
 * through Marketplace → Capabilities → Import from GitHub, which the local
 * operator can use because the operator is the instance's super admin. The
 * skills repo is the source of truth for the library; re-import to pick up
 * updates.
 *
 * The URL is the plain repository page, not the `.git` clone URL: people copy
 * and paste what they see, and the browser address bar has no `.git`. The
 * import route appends `.git` itself when it is missing.
 */
export const BASELINE_SKILLS_REPO_URL = 'https://github.com/AutomatosAI/automatos-skills'

/** The same repo without the `.git` suffix, for display. */
export const BASELINE_SKILLS_REPO_LABEL = 'github.com/AutomatosAI/automatos-skills'
