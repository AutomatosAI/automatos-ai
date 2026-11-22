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
| Deployment `<release>-api` | The API. It starts uvicorn with `AUTOMATOS_SKIP_MIGRATIONS=true`, so the image's entrypoint runs no boot step and the Job is the only thing that changes the schema. **One replica**: the in-process schedulers use a per-container lock, so a second replica would run every scheduled job twice. |
| Deployment `<release>-frontend` | The web app. |
| Deployment `<release>-worker` | The workspace worker (Code Canvas runtime). One pod, `Recreate` strategy. Its entrypoint starts as root with four capabilities (`CHOWN`, `DAC_OVERRIDE`, `SETGID`, `SETUID`) to fix the volume's ownership, then drops to uid 1000 with no capabilities; nothing in the container can gain privileges (`allowPrivilegeEscalation: false`). |
| ServiceAccounts `<fullname>-api`, `<fullname>-frontend`, `<fullname>-worker`, `<fullname>-migrate` | One account for each component, so a cloud role you tie to one reaches that component only (`<fullname>` is the prefix the chart puts on every resource's name: the release name, with `-automatos` added unless the release name already contains it (release `demo`: `demo-automatos`)). The migration Job's account is a hook that runs just before the Job. Helm deletes it once the hooks succeed; after a failed Job it can be left behind until the next install or upgrade replaces it. None of them mounts a token for the cluster's own API: nothing in Automatos calls it. |
| PVC `<release>-workspaces` | Agent files: the worker reads and writes, the API reads. Kept on `helm uninstall`. Pods on several nodes need `ReadWriteMany` storage. |
| PVC `<release>-api-data` | The API's files at `/app/data`: uploads, brand files and generated documents. Kept on `helm uninstall`. With `ReadWriteOnce`, the default, every rollout stops the old API pod first, so the API is down until the new pod is ready. |
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
   | `REDIS_URL` | `redis://` is a plain connection. A Redis that requires TLS takes `rediss://`, and the server's certificate and host name are then checked against the image's trust store; nothing in the address can switch that check off. A user name (a Redis ACL user) goes before the password: `rediss://USER:PASSWORD@HOST:6379/0`. |
   | `API_KEY` | The platform API key. |
   | `CREDENTIAL_ENCRYPTION_KEY` | A Fernet key. **Keep it and back it up**: credentials saved in the app are encrypted with it, and every pod must share it. |
   | `WORKER_INTERNAL_TOKEN` | Optional. Shared secret between the API and the worker. |
   | `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` | Optional. For the worker's Code Canvas agent. |
   | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Optional. Object storage credentials, for S3 or for MinIO with `S3_ENDPOINT_URL` in `config`. |

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

LLM keys are added in the app (Settings → Model keys) and stored encrypted in the
database.

## Settings

See [`values.yaml`](values.yaml). The ones you're most likely to change:

| Value | Default | |
|---|---|---|
| `existingSecret` | *(required)* | The Secret above. |
| `edition.defaultWorkspaceId` | `00000000-…-0000000000c1` | The local workspace's id. |
| `config` | optional services off | Extra non-secret settings for the API and the Job: a name `orchestrator/config.py` reads, e.g. `S3_ENDPOINT_URL`, `QDRANT_URL`. Rendering fails on `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (they go in the Secret), on `DOCUMENT_UPLOAD_DIR` and `DOCUMENT_STORAGE_DIR` while `api.data` is on (and, while it is off under a read-only root, on either outside the API's scratch volumes), on `HOME` under the API's read-only root (the chart sets it), and on `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT` (set `database.*`). Quote `LOCAL_REMOTE_ACCESS`: YAML reads a bare `off` as `false`, which the API refuses. |
| `database.sslmode`, `database.caConfigMap`, `database.caKey` | empty, empty, `ca.crt` | TLS on every connection to PostgreSQL, from the API, the migration Job and the worker. `sslmode` sets `DATABASE_SSLMODE`; `verify-full` checks the server's certificate and its host name. `caConfigMap` names a ConfigMap you create in the release's namespace holding the CA that signed the server's certificate (for a managed database, the provider's bundle), under the key `caKey`: the chart mounts it read-only at `/etc/automatos/database-ca` in all three and sets `DATABASE_SSLROOTCERT` to the file. It needs `sslmode` `verify-ca` or `verify-full`. When you change the ConfigMap, the kubelet updates the mounted file after its sync delay (a minute or so); each new connection then trusts the new file, while a connection already open keeps its session until it closes, and a file with no certificate in it stops a pod at its next start. Keep the old and the new CA in one bundle until the server presents its new certificate, and restart the pods to be certain. All empty: nothing is set or mounted, and each connection behaves as before. Example: `kubectl -n automatos create configmap database-ca --from-file=ca.crt=<bundle>.pem`, then `--set database.sslmode=verify-full --set database.caConfigMap=database-ca`. |
| `api.image.tag`, `frontend.image.tag`, `worker.image.tag` | the chart's `appVersion` (`edge`) | Pin a `sha-…` or version tag for repeatable installs (below). |
| `api.image.digest`, `frontend.image.digest`, `worker.image.digest` | empty | `sha256:<64 hex>`: the pod runs `repository@digest`, that exact build, whatever the tag. |
| `api.image.pullPolicy`, `frontend.image.pullPolicy`, `worker.image.pullPolicy` | empty: the chart chooses | `Always` for a tag that moves (`edge`, or a `MAJOR.MINOR` release line), `IfNotPresent` for a digest or a fixed tag. A value you set is used as given. |
| `workspaces.storageClass`, `workspaces.accessModes` | cluster default, `ReadWriteMany` | |
| `api.data.enabled`, `api.data.size`, `api.data.storageClass`, `api.data.accessModes` | on, `10Gi`, cluster default, `ReadWriteOnce` | The API's files. Off, they're lost on every restart. `accessModes` can't change once the claim exists; more than one API replica needs a mode other than `ReadWriteOnce` or `ReadWriteOncePod`. The storage must honour the pod's `fsGroup`, or the API can't write here: some NFS-style drivers don't. |
| `serviceAccount.create` | `true` | The chart creates one account for each component (above). `false`: it creates none, and each pod runs under the account `serviceAccount.<component>.name` names, or the namespace's `default`; several components may name the same one. |
| `serviceAccount.api`, `.frontend`, `.worker`, `.migrate` (`.name`, `.annotations`) | `<fullname>-<component>`, none | `name` renames that account; `annotations` go on it alone, e.g. `eks.amazonaws.com/role-arn` or `iam.gke.io/gcp-service-account` on the API's. With `serviceAccount.create=true` each account needs a name of its own. Tying a role to an account is yours to do; a token a cloud's webhook projects for that role (EKS IRSA) is separate from the cluster token the chart turns off. |
| `api.readOnlyRootFilesystem`, `frontend.readOnlyRootFilesystem`, `worker.readOnlyRootFilesystem`, `migrations.readOnlyRootFilesystem` | `true` | A read-only root file system for that component's container, with an `emptyDir` at each path it writes outside its volumes (below). `false`: the root stays writable and nothing is mounted for it, as before chart 0.9.0. |
| `ingress.*` | off | |
| `otel.enabled`, `otel.endpoint` | off | OpenTelemetry traces from the API, the worker and the web app's server side, and the API's GenAI metrics, over OTLP/HTTP to your collector ([`deploy/otel/`](../../deploy/otel/) has a reference config). Headers for the endpoint go in the Secret as `OTEL_EXPORTER_OTLP_HEADERS`. `otel.samplerRatio` keeps a share of new traces; `otel.resourceAttributes` adds to the pod and namespace names on every span. |
| `sessionMode.enabled` | off | Session mode: your own Claude Code, Codex or Copilot sessions as agents, through the CLI host on your machine. Local edition only. Each session's files are uploaded into the workspace volume; the projects folder isn't browsable on a cluster. With the ingress on, `/api/v1/cli-hosts` gets its own Ingress (`sessionMode.ingressAnnotations`, a 64 MB body limit for ingress-nginx). |

## Pod security

Every container runs with the runtime's default system-call filter (`seccompProfile: RuntimeDefault`), cannot gain privileges (`allowPrivilegeEscalation: false`) and drops every capability. The API, the web app and the migration Job also run as a non-root user. The worker is the exception until a later release: it starts as root and adds back the four capabilities in the table above, so it passes the "baseline" pod security level, not "restricted".

Every container's root file system is read-only, so what runs stays what was released. Each component writes only to its volumes and to the scratch volumes below: `emptyDir`s, on the node's disk, emptied when the pod is replaced. [`docs/PRDS/PRD-280-US-103-WRITE-PATHS.md`](../../docs/PRDS/PRD-280-US-103-WRITE-PATHS.md) lists what writes each path, file and line.

| Component | Read-only root | Scratch volumes |
|---|---|---|
| API | yes | `/tmp`, `/home/automatos` (its `HOME`, which the chart sets), `/app/logs`; with `api.data.enabled=false`, `/app/documents` too |
| Migration Job | yes | `/tmp` |
| Web app | yes | none: its server writes no file |
| Worker | yes | `/tmp`, `/home/worker` (its `HOME`) |

Each component's `readOnlyRootFilesystem` value turns it off for that component alone. A path of your own in `config` (a temporary folder, a log file) has to be under one of these, or the API cannot write it.

CI installs the chart into a kind cluster on every change to the chart (Markdown aside) or to an image's start-up (`.github/workflows/chart.yml`, `deploy/kind/strict.sh`), with the values in `deploy/kind/strict-values.yaml`, whose header lists what they turn on and what they do not exercise. Every pod must become ready. Every pod the release makes, the migration Job's included, must be admitted by a namespace that enforces "restricted"; the worker, as above, may be refused only for the violations `strict.sh` lists. Every container and init container of those pods must set `readOnlyRootFilesystem: true` in its spec, and in the API, the web app and the worker, which keep running, a write to the root must fail.

## Upgrading to 0.9.0

- **Every container's root file system becomes read-only** (`readOnlyRootFilesystem: true`), with the scratch volumes in the table above. Anything a pod wrote on its own disk outside them now fails with "read-only file system". To keep a component as it was, set its value to `false` (`--set api.readOnlyRootFilesystem=false`, and the same for `frontend`, `worker` and `migrations`).
- **The API's home folder, `/tmp` and `/app/logs` start empty on every new pod,** as the container's own disk already did. The API's pods now set `HOME=/home/automatos`.
- **Values that rendered on 0.8.0 and now stop the render,** with a message that names the value and the way out:
  - with `api.data.enabled=false`, a `DOCUMENT_UPLOAD_DIR` or `DOCUMENT_STORAGE_DIR` in `config` that is not under one of the API's scratch volumes (`/tmp`, `/home/automatos`, `/app/logs`, `/app/documents`; a relative path is under `/app`). Fix it with a path under one of them (`/tmp/uploads`, `/app/documents`), or with `api.readOnlyRootFilesystem=false`;
  - `HOME` in `config`, which the chart now sets for the API. Remove it, or set `api.readOnlyRootFilesystem=false`.
- **Your own paths in `config`** (`WORKSPACE_EXPORT_TEMP_DIR`, `AUTOMATOS_WORKSPACE`, `DECISION_SHADOW_LOG_PATH` and the like) must be under a scratch volume or a volume, or set `api.readOnlyRootFilesystem=false`. The chart checks only the two document paths.
- **The migration Job gets an `emptyDir` at `/tmp`,** and the worker one at `/tmp` and one at `/home/worker`.
- **Start-up can take longer.** Python cannot write compiled files (`__pycache__`) to a read-only root, so a module the image does not ship compiled is compiled in memory at every start. The API's image ships its libraries and its start-up modules compiled, and its start-up probe allows five minutes (60 checks, 5 s apart); the worker's image ships its libraries compiled, not its own code.
- **`helm upgrade --reuse-values` does not turn the read-only roots on.** It renders with the values of the release as it was, the old chart's defaults included, so the four new values are unset and each root stays writable as on 0.8.0. `--reset-then-reuse-values` (Helm 3.14 and later) takes the new chart's defaults and then your release's values over them, so the roots become read-only. To be certain either way, set the four values explicitly: `--set api.readOnlyRootFilesystem=true --set frontend.readOnlyRootFilesystem=true --set worker.readOnlyRootFilesystem=true --set migrations.readOnlyRootFilesystem=true` (or `false`).

## Upgrading to 0.7.0

- **Each component gets its own account,** `<fullname>-api`, `-frontend`, `-worker` and `-migrate`, in place of the one `<fullname>` account the API, the web app and the worker shared (`<fullname>` is the prefix the chart puts on every resource's name: the release name, with `-automatos` added unless the release name already contains it (release `demo`: `demo-automatos`)). The upgrade creates the new accounts and deletes the old one.
- **`serviceAccount.name` and `serviceAccount.annotations` are gone.** Rendering stops with a message if either is set: put the name or the annotations on the account that needs them, e.g. `serviceAccount.api.annotations`, and with `serviceAccount.create=false` name your account under each component (`serviceAccount.api.name`, `.frontend.name`, `.worker.name`, `.migrate.name`; they may all name the same one).
- **A cloud role trusts an account by its name** (an IRSA trust policy's subject, an EKS Pod Identity association, a GKE or AKS federated identity, a RoleBinding), so one bound to the old `<fullname>` account now reaches nothing. Bind it to the new account, or keep the old name for the component that needs it: `--set serviceAccount.api.name=<fullname>` (release `demo`: `--set serviceAccount.api.name=demo-automatos`). EKS Pod Identity needs no annotation, so for it rendering does not stop: check before you upgrade.
- **The API no longer mounts a cluster token,** and the migration Job now runs under `<fullname>-migrate` rather than the namespace's `default` account.
- **Every container gets the default system-call filter.** The worker already ran under it. In the API, the web app and the migration Job, a system call the runtime's default profile blocks now fails with an error.

## Upgrading to 0.6.0

- **Move object storage credentials to the Secret.** With `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` in `config`, rendering now stops with a message. Add them to the Secret under the same keys and remove them from `config`.
- **The upgrade creates a 10 Gi claim,** so the cluster needs a default storage class, or set `api.data.storageClass`. With `api.data.enabled=false` the API's files stay on the container's disk.
- **Files already on an API pod's own disk aren't copied** to the new volume. Copy out any you need before upgrading.
- **More than one API replica** needs `api.data.accessModes` set first, or `api.data.enabled=false`. Rendering stops otherwise.

## Pinning the images

`edge`, the default, moves with every main publish and every weekly OS rebuild, so
the chart pulls it `Always`: with `IfNotPresent` each node would keep whichever
`edge` it pulled first, and two pods could run different builds. For a repeatable
install, pin one build instead, and the chart pulls it `IfNotPresent`:

- **A fixed tag:** `--set api.image.tag=sha-1a2b3c4` (and the same for `frontend`
  and `worker`). A `sha-<7>` tag is never re-pushed.
- **A digest:** `--set api.image.digest=sha256:<64 hex>`. The pod runs
  `repository@digest`; `crane digest` or `docker buildx imagetools inspect` prints
  a tag's digest.
- **At package time:** `helm package charts/automatos --app-version sha-1a2b3c4`
  stamps the chart's `appVersion`, the tag every component uses when its own tag
  and digest are empty.

## Tests

- **Unit tests, no cluster:** `helm unittest charts/automatos` (the
  [helm-unittest](https://github.com/helm-unittest/helm-unittest) plugin). CI runs
  them with `helm lint` and a kubeconform schema check
  (`.github/workflows/chart.yml`).
- **An install under a strict namespace, in CI:** `chart.yml`'s `install` job builds
  the three images, installs the chart into a kind cluster with the values in
  `deploy/kind/strict-values.yaml` (its header lists what is on and what is not
  exercised), and checks the pod security level and the read-only roots (above), with
  `deploy/kind/strict.sh`.
- **End to end, on a throwaway kind cluster:** `deploy/kind/e2e.sh cycle 2`
  creates a cluster, loads the images (the API built from your checkout), installs
  with dev PostgreSQL and Redis, runs the checks, upgrades, runs them again and
  deletes the cluster, twice. It also runs an OpenTelemetry Collector on the
  reference config and checks that one request's spans from the API and the worker
  arrive as one trace. It needs Docker, kind, kubectl and helm, and several
  GB of images, so it runs locally, not in CI. `e2e.sh up` keeps the cluster for
  poking at; `e2e.sh down` removes it.
