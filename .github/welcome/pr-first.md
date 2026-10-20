Welcome @{login}, and thanks for your first pull request here! 🎉 A few things that help it land quickly:

- **CI is the gate.** It runs on this pull request; keep an eye on the checks. Red is red, even on a lane marked non-required.
- **Tests.** Every fix ships with a test that fails without it. [AGENTS.md]({repo_url}/blob/main/AGENTS.md) has the rules CI checks on the lines you change (functions of 50 lines or fewer, nesting of 4 levels at most); if you use an AI coding agent, point it at that file first.
- **Sign-off.** Every commit needs a `Signed-off-by:` trailer (`git commit -s`), or the `dco` check fails. Missed one? `git commit --amend -s --no-edit` fixes the last commit and `git rebase --signoff HEAD~N` the last N; then `git push --force-with-lease` to your branch.
- **The PR template.** Fill it in honestly: what changed and why, what you actually ran, and the AI-assistance disclosure (which tool, and what it wrote).
- **Link the issue** with `Closes #N` in the description, so it closes when this merges.
- **Review happens here,** on the pull request. CodeRabbit leaves automated review comments, and a maintainer reviews too.

Ask here if anything's unclear. And if Automatos is useful to you, a ⭐ on the repo helps more people find it.

---
_This reply is automated. A maintainer reads every pull request._
