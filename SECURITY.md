# Security policy

## Reporting a vulnerability

**Please don't open a public issue for a security problem.** Report it privately instead:

1. **Preferred:** use GitHub's private vulnerability reporting, under the repository's **Security** tab → **Report a vulnerability**.
2. **Or email** support@automatos.ai with `SECURITY` at the start of the subject line.

Please include:
- what you found, and where (the file, route or feature);
- the steps to reproduce it, or a proof of concept;
- what an attacker could do with it;
- which edition you tested: local (`AUTH_EDITION=local`) or hosted.

We acknowledge every report, keep you informed until it's resolved, and credit you in the fix if you'd like us to. Please give us reasonable time to fix a problem before you disclose it.

## Supported versions

Security fixes land on `main` and ship in the next release of both editions. Older tags are not patched separately.

## Scope

In scope:
- the code in this repository: the orchestrator, the frontend, the services, and the Docker and compose configuration;
- the hosted edition's behaviour where it comes from this code.

Out of scope:
- vulnerabilities in third-party services or dependencies, unless our use of them is the problem. Report those upstream;
- denial of service by volume;
- social engineering.

## If a secret is exposed

If a credential, token or key ever lands in the repository, a commit, a log or an issue:

1. **Treat it as burned.** Rotate it at the provider first, then remove it. Removing it from the history does not make it safe again.
2. **Say where it was,** so every copy can be traced.
3. **Check for the same pattern elsewhere** before continuing.

gitleaks scans every push. A false positive in a test fixture needs a fingerprint entry, not a disabled check.
