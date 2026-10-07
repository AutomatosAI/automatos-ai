# Automatos AI Helm chart

Installs Automatos AI on Kubernetes: the API, a database migration Job, the web
app and the workspace worker. PostgreSQL (with the `pgvector` extension) and
Redis are **external**: use managed services, or anything you run yourself.

> **Local edition only, for now.** The published web-app image is the local
> edition: one workspace, one operator, **no login**. Anyone who can reach the web
> app is the admin, so put it behind a VPN or an authenticating ingress. The web
> app also has its API address fixed at build time (`http://localhost:8000`), so
> today you reach it with `kubectl port-forward` (below). Both change in later
> releases.

## What it deploys

| Resource | What it does |
|---|---|
| Job `<release>-migrate` | A `pre-install`/`pre-upgrade` hook running `docker-entrypoint.sh migrate`: waits for PostgreSQL, builds an empty database, migrates, seeds, and (local edition) creates the workspace and operator. A failure fails the install or upgrade. |
| Deployment `<release>-api` | The API. It starts uvicorn directly, so the Job is the only thing that changes the schema. **One replica**: the in-process schedulers use a per-container lock, so a second replica would run every scheduled job twice. |
| Deployment `<release>-frontend` | The web app. |
| Deployment `<release>-worker` | The workspace worker (Code Canvas runtime). One pod, `Recreate` strategy. |
| PVC `<release>-workspaces` | Agent files: the worker reads and writes, the API reads. Kept on `helm uninstall`. Pods on several nodes need `ReadWriteMany` storage. |
| Ingress (off by default) | `/api` and `/ws` to the API, everything else to the web app. Useful once the web app reads its API address at runtime. |

## Install

1. Create the Secret. The chart never generates or stores secrets:

   ```bash
   kubectl create namespace automatos
   kubectl -n automatos create secret generic automatos-secrets \
     --from-literal=DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DB' \
     --from-literal=REDIS_URL='redis://:PASSWORD@HOST:6379/0' \
     --from-literal=API_KEY="$(openssl rand -hex 24)" \
     --from-literal=CREDENTIAL_ENCRYPTION_KEY="$(python3 -c 'import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')" \
     --from-literal=WORKER_INTERNAL_TOKEN="$(openssl rand -hex 24)"
   ```

   | Key | |
   |---|---|
   | `DATABASE_URL` | PostgreSQL with `pgvector`. |
   | `REDIS_URL` | |
   | `API_KEY` | The platform API key. |
   | `CREDENTIAL_ENCRYPTION_KEY` | A Fernet key. **Keep it and back it up**: credentials saved in the app are encrypted with it, and every pod must share it. |
   | `WORKER_INTERNAL_TOKEN` | Optional. Shared secret between the API and the worker. |
   | `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` | Optional. For the worker's Code Canvas agent. |

2. Install:

   ```bash
   helm install automatos charts/automatos -n automatos \
     --set existingSecret=automatos-secrets --wait --timeout 20m
   ```

3. Reach it (see the note at the top):

   ```bash
   kubectl -n automatos port-forward svc/automatos-api 8000:8000 &
   kubectl -n automatos port-forward svc/automatos-frontend 3000:3000 &
   open http://localhost:3000
   ```

LLM keys are added in the app (Settings → API Keys) and stored encrypted in the
database.

## Settings

See [`values.yaml`](values.yaml). The ones you're most likely to change:

| Value | Default | |
|---|---|---|
| `existingSecret` | *(required)* | The Secret above. |
| `edition.defaultWorkspaceId` | `00000000-…-0000000000c1` | The local workspace's id. |
| `config` | optional services off | Extra non-secret settings for the API and the Job: any name `orchestrator/config.py` reads, e.g. `S3_ENDPOINT_URL`, `QDRANT_URL`. |
| `api.image.tag`, `frontend.image.tag`, `worker.image.tag` | the chart's `appVersion` (`edge`) | Pin a `sha-…` or version tag for repeatable installs. |
| `workspaces.storageClass`, `workspaces.accessModes` | cluster default, `ReadWriteMany` | |
| `ingress.*` | off | |
| `sessionMode.enabled` | off | Session mode: your own Claude Code, Codex or Copilot sessions as agents, through the CLI host on your machine. Local edition only. Each session's files are uploaded into the workspace volume; the projects folder isn't browsable on a cluster. With the ingress on, `/api/v1/cli-hosts` gets its own Ingress (`sessionMode.ingressAnnotations`, a 64 MB body limit for ingress-nginx). |

## Tests

- **Unit tests, no cluster:** `helm unittest charts/automatos` (the
  [helm-unittest](https://github.com/helm-unittest/helm-unittest) plugin). CI runs
  them with `helm lint` and a kubeconform schema check
  (`.github/workflows/chart.yml`).
- **End to end, on a throwaway kind cluster:** `deploy/kind/e2e.sh cycle 2`
  creates a cluster, loads the images (the API built from your checkout), installs
  with dev PostgreSQL and Redis, runs the checks, upgrades, runs them again and
  deletes the cluster, twice. It needs Docker, kind, kubectl and helm, and several
  GB of images, so it runs locally, not in CI. `e2e.sh up` keeps the cluster for
  poking at; `e2e.sh down` removes it.
