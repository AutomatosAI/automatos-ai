# Automatos AI Helm chart

Installs Automatos AI on Kubernetes: the API, a database migration Job, the web
app and the workspace worker. PostgreSQL (with the `pgvector` extension) and
Redis are **external**: use managed services, or anything you run yourself.

> **Two editions.** The default, `local`, is one workspace, one operator and **no
> login**: anyone who can reach the web app is the admin, so put it behind a VPN or
> an authenticating ingress. Its web app has its API address fixed at build time
> (`http://localhost:8000`), so you reach it with `kubectl port-forward` (below).
> `enterprise` signs people in with your organisation's identity provider and needs
> the ingress, so that the web app and the API answer on one host: see
> [The enterprise edition](#the-enterprise-edition).

## What it deploys

| Resource | What it does |
|---|---|
| Job `<release>-migrate` | A `pre-install`/`pre-upgrade` hook running `docker-entrypoint.sh migrate`: waits for PostgreSQL, builds an empty database, migrates, seeds, and (local edition) creates the workspace and operator. A failure fails the install or upgrade. |
| Deployment `<release>-api` | The API. It starts uvicorn with `AUTOMATOS_SKIP_MIGRATIONS=true`, so the image's entrypoint runs no boot step and the Job is the only thing that changes the schema. **One replica**: the in-process schedulers use a per-container lock, so a second replica would run every scheduled job twice. |
| Deployment `<release>-frontend` | The web app: the edition's own image, `frontend.image` (local) or `frontend.enterpriseImage` (enterprise), as `edition.authEdition` names. |
| Deployment `<release>-worker` | The workspace worker (Code Canvas runtime). One pod, `Recreate` strategy. It runs as uid 1000 with no capability, and nothing in the container can gain privileges (`allowPrivilegeEscalation: false`). It re-owns nothing: the storage must hand its volume over writable by the pod's group (`fsGroup: 1000`), and it stops with a message that says so when it cannot write the volume. |
| ServiceAccounts `<fullname>-api`, `<fullname>-frontend`, `<fullname>-worker`, `<fullname>-migrate` | One account for each component, so a cloud role you tie to one reaches that component only (`<fullname>` is the prefix the chart puts on every resource's name: the release name, with `-automatos` added unless the release name already contains it (release `demo`: `demo-automatos`)). The migration Job's account is a hook that runs just before the Job. Helm deletes it once the hooks succeed; after a failed Job it can be left behind until the next install or upgrade replaces it. None of them mounts a token for the cluster's own API: nothing in Automatos calls it. |
| PVC `<release>-workspaces` | Agent files: the worker reads and writes, the API reads. Kept on `helm uninstall`. Pods on several nodes need `ReadWriteMany` storage. Not created when either side names a claim of its own (`workspaces.api.existingClaim`, `workspaces.worker.existingClaim`). |
| PVC `<release>-api-data` | The API's files at `/app/data`: uploads, brand files and generated documents. Kept on `helm uninstall`. With `ReadWriteOnce`, the default, every rollout stops the old API pod first, so the API is down until the new pod is ready. |
| Ingress (off by default) | `/api`, `/ws` and exactly `/.well-known/automatos.json` (the discovery document a phone reads) to the API, everything else to the web app, its own `/.well-known/` files among it. `/exports/`, `/health`, `/metrics` and `/docs` are not routed. The enterprise edition needs it; the local web app calls `http://localhost:8000`, fixed at build time. |

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
   | `WORKER_INTERNAL_TOKEN` | Optional. Shared secret between the API and the worker. Required, in this Secret and the worker's, when the worker has a Secret of its own (`worker.existingSecret`). |
   | `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` | Optional. For the worker's Code Canvas agent. |
   | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Optional. Object storage credentials, for S3 or for MinIO with `S3_ENDPOINT_URL` in `config`. |
   | `OIDC_CLIENT_SECRET` | The enterprise edition only, and required there: the application's client secret at your identity provider. |

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

## Install a released version

Each release (a `vMAJOR.MINOR.PATCH` tag) publishes this chart to GHCR as an OCI
artifact, with its `version` and `appVersion` set to the release's version and every
image named by the digest that release published and signed, the web image of each
published edition included (`frontend.image` for local, `frontend.enterpriseImage` for
enterprise). No checkout is needed:

```bash
helm install automatos oci://ghcr.io/automatosai/charts/automatos --version 0.7.0 \
  -n automatos --set existingSecret=automatos-secrets --wait --timeout 20m
```

A version is released when its release run (the `images` workflow on the tag) has
finished green. Check the chart's and the images' signatures with the command in
[`docs/deployment-infrastructure/releases.md`](../../docs/deployment-infrastructure/releases.md#checking-a-signature)
before you trust them. Name the version: a release never replaces a published
version, so the same version always installs the same images. The registry does not
enforce that; the release workflow's checks and the repository's tag ruleset do. The
same page covers release candidates and what a run that fails part-way leaves behind.

## Supported Kubernetes versions

CI renders the chart and checks it against the schemas of Kubernetes **1.31** (the
oldest it supports) and **1.37** (the newest), on every change to the chart
(`.github/workflows/chart.yml`).

## Settings

See [`values.yaml`](values.yaml). The ones you're most likely to change:

| Value | Default | |
|---|---|---|
| `existingSecret` | *(required)* | The Secret above. |
| `edition.authEdition` | `local` | Which edition the API and the web app run: `local`, `enterprise` (below) or `saas` (the hosted edition), spelled exactly so. Rendering stops on any other spelling. |
| `edition.defaultWorkspaceId` | `00000000-…-0000000000c1` | The local workspace's id; in enterprise, the default workspace's (a UUID). |
| `edition.enterprise.*` | empty | The enterprise edition's sign-in: [The enterprise edition](#the-enterprise-edition). |
| `config` | optional services off | Extra non-secret settings for the API and the Job: a name `orchestrator/config.py` reads, e.g. `S3_ENDPOINT_URL`, `QDRANT_URL`. Rendering fails on `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (they go in the Secret), on `DOCUMENT_UPLOAD_DIR` and `DOCUMENT_STORAGE_DIR` while `api.data` is on (and, while it is off under a read-only root, on either outside the API's scratch volumes), on `HOME` under the API's read-only root (the chart sets it), on `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT` (set `database.*`), on `OIDC_CLIENT_SECRET` (it goes in the Secret), and in the enterprise edition on each setting `edition.enterprise` sets and on `SESSION_COOKIE_NAME` and `SESSION_CSRF_HEADER`. Quote `LOCAL_REMOTE_ACCESS`: YAML reads a bare `off` as `false`, which the API refuses. |
| `database.sslmode`, `database.caConfigMap`, `database.caKey` | empty, empty, `ca.crt` | TLS on every connection to PostgreSQL, from the API, the migration Job and the worker. `sslmode` sets `DATABASE_SSLMODE`; `verify-full` checks the server's certificate and its host name. `caConfigMap` names a ConfigMap you create in the release's namespace holding the CA that signed the server's certificate (for a managed database, the provider's bundle), under the key `caKey`: the chart mounts it read-only at `/etc/automatos/database-ca` in all three and sets `DATABASE_SSLROOTCERT` to the file. It needs `sslmode` `verify-ca` or `verify-full`. When you change the ConfigMap, the kubelet updates the mounted file after its sync delay (a minute or so); each new connection then trusts the new file, while a connection already open keeps its session until it closes, and a file with no certificate in it stops a pod at its next start. Keep the old and the new CA in one bundle until the server presents its new certificate, and restart the pods to be certain. All empty: nothing is set or mounted, and each connection behaves as before. Example: `kubectl -n automatos create configmap database-ca --from-file=ca.crt=<bundle>.pem`, then `--set database.sslmode=verify-full --set database.caConfigMap=database-ca`. |
| `api.image.tag`, `frontend.image.tag`, `frontend.enterpriseImage.tag`, `worker.image.tag` | the chart's `appVersion` (`edge`) | Pin a `sha-…` or version tag for repeatable installs (below). |
| `api.image.digest`, `frontend.image.digest`, `frontend.enterpriseImage.digest`, `worker.image.digest` | empty | `sha256:<64 hex>`: the pod runs `repository@digest`, that exact build, whatever the tag. |
| `api.image.pullPolicy`, `frontend.image.pullPolicy`, `frontend.enterpriseImage.pullPolicy`, `worker.image.pullPolicy` | empty: the chart chooses | `Always` for a tag that moves (`edge`, or a `MAJOR.MINOR` release line), `IfNotPresent` for a digest or a fixed tag. A value you set is used as given. |
| `frontend.enterpriseImage.repository` | `ghcr.io/automatosai/automatos-enterprise-frontend` | The enterprise edition's web image, which the chart runs in place of `frontend.image` when `edition.authEdition` is `enterprise`. Each web image is built for one edition. |
| `frontend.mediaOrigins` | empty | Enterprise only: the origins the browser may load media from (your object storage's public host), which the web server adds to its content security policy when it starts (`MEDIA_ORIGINS`). Up to 16, separated by commas or spaces, each `https://<host>[:port]` or `https://*.<domain>.<suffix>[:port]`. Name the narrowest host that works, your storage's own public host, rather than a wildcard. Rendering fails on a value the server would refuse, and on any value in another edition. |
| `workspaces.storageClass`, `workspaces.accessModes` | cluster default, `ReadWriteMany` | |
| `workspaces.api.existingClaim`, `workspaces.worker.existingClaim` | empty | A claim that already exists for that side of the agent-files volume; the chart then creates none. In one namespace, a claim named for either side serves both. With `worker.namespace`, both are required (below). |
| `worker.namespace` | empty: the release's namespace | The worker's Deployment, Service and account go there, and the API's `WORKER_INTERNAL_URL` becomes `http://<fullname>-worker.<namespace>.svc:8081`. The release's own namespace counts as empty. See "The worker in a namespace of its own" below. |
| `worker.existingSecret` | empty: `existingSecret` | The worker's own Secret, in the worker's namespace. The worker reads no other: `DATABASE_URL`, `REDIS_URL` and `WORKER_INTERNAL_TOKEN` (required), `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` and `OTEL_EXPORTER_OTLP_HEADERS` (optional). Set, `WORKER_INTERNAL_TOKEN` becomes required in `existingSecret` too, the same value. |
| `api.data.enabled`, `api.data.size`, `api.data.storageClass`, `api.data.accessModes` | on, `10Gi`, cluster default, `ReadWriteOnce` | The API's files. Off, they're lost on every restart. `accessModes` can't change once the claim exists; more than one API replica needs a mode other than `ReadWriteOnce` or `ReadWriteOncePod`. The storage must honour the pod's `fsGroup`, or the API can't write here: some NFS-style drivers don't. |
| `serviceAccount.create` | `true` | The chart creates one account for each component (above). `false`: it creates none, and each pod runs under the account `serviceAccount.<component>.name` names, or the namespace's `default`; several components may name the same one. |
| `serviceAccount.api`, `.frontend`, `.worker`, `.migrate` (`.name`, `.annotations`) | `<fullname>-<component>`, none | `name` renames that account; `annotations` go on it alone, e.g. `eks.amazonaws.com/role-arn` or `iam.gke.io/gcp-service-account` on the API's. With `serviceAccount.create=true` each account needs a name of its own. Tying a role to an account is yours to do; a token a cloud's webhook projects for that role (EKS IRSA) is separate from the cluster token the chart turns off. |
| `api.readOnlyRootFilesystem`, `frontend.readOnlyRootFilesystem`, `worker.readOnlyRootFilesystem`, `migrations.readOnlyRootFilesystem` | `true` | A read-only root file system for that component's container, with an `emptyDir` at each path it writes outside its volumes (below). `false`: the root stays writable and nothing is mounted for it, as before chart 0.9.0. |
| `api`, `frontend`, `worker`, `migrations` (`.nodeSelector`, `.tolerations`, `.affinity`, `.topologySpreadConstraints`) | empty: anywhere | Where that component's pods may run, each rendered as given into its pod spec. For the worker on nodes of its own: `worker.nodeSelector` for their label and `worker.tolerations` for their taint. The memory store is not in the chart yet (PRD-280 Wave 6); it gets the same four when it is. |
| `ingress.*` | off | |
| `otel.enabled`, `otel.endpoint` | off | OpenTelemetry traces from the API, the worker and the web app's server side, and the API's GenAI metrics, over OTLP/HTTP to your collector ([`deploy/otel/`](../../deploy/otel/) has a reference config). Headers for the endpoint go in the Secret as `OTEL_EXPORTER_OTLP_HEADERS`. `otel.samplerRatio` keeps a share of new traces; `otel.resourceAttributes` adds to the pod and namespace names on every span. |
| `sessionMode.enabled` | off | Session mode: your own Claude Code, Codex or Copilot sessions as agents, through the CLI host on your machine. Local edition only. Each session's files are uploaded into the workspace volume; the projects folder isn't browsable on a cluster. With the ingress on, `/api/v1/cli-hosts` gets its own Ingress (`sessionMode.ingressAnnotations`, a 64 MB body limit for ingress-nginx). |

## Pod security

Every container runs with the runtime's default system-call filter (`seccompProfile: RuntimeDefault`), cannot gain privileges (`allowPrivilegeEscalation: false`), drops every capability and adds none, and runs as a user other than root: the API, the migration Job and the worker as uid 1000, the web app as uid 1001. Every pod passes the "restricted" pod security level.

The API's data volume and the workspaces volume must be storage that honours the pod's `fsGroup` (1000), so uid 1000 can write them; some NFS-style drivers do not. A worker that cannot write its volume stops at start, and its log says what the storage must provide. The worker's pod sets `fsGroupChangePolicy: OnRootMismatch`: the kubelet changes the volume's group and permissions only when its top folder does not already match, so a restart does not walk every agent file on a large volume.

Every container's root file system is read-only, so what runs stays what was released. Each component writes only to its volumes and to the scratch volumes below: `emptyDir`s, on the node's disk, emptied when the pod is replaced. [`docs/PRDS/PRD-280-US-103-WRITE-PATHS.md`](../../docs/PRDS/PRD-280-US-103-WRITE-PATHS.md) lists what writes each path, file and line.

| Component | Read-only root | Scratch volumes |
|---|---|---|
| API | yes | `/tmp`, `/home/automatos` (its `HOME`, which the chart sets), `/app/logs`; with `api.data.enabled=false`, `/app/documents` too |
| Migration Job | yes | `/tmp` |
| Web app | yes | none: its server writes no file |
| Worker | yes | `/tmp`, `/home/worker` (its `HOME`) |

Each component's `readOnlyRootFilesystem` value turns it off for that component alone. A path of your own in `config` (a temporary folder, a log file) has to be under one of these, or the API cannot write it.

CI installs the chart into a kind cluster on every change to the chart (Markdown aside) or to an image's start-up (`.github/workflows/chart.yml`, `deploy/kind/strict.sh`), with the values in `deploy/kind/strict-values.yaml`, whose header lists what they turn on and what they do not exercise. The worker runs there in a namespace of its own, with a Secret of its own and the agent-files volume as two claims on one folder, as "The worker in a namespace of its own" below describes; the release's namespace and the worker's both enforce "restricted", and every pod must become ready. The worker must run as uid 1000 and write its volume, and the API must read what it wrote through its own claim. Every pod the release makes, the migration Job's included, must also be admitted by a second namespace that enforces "restricted", as a dry run, so a refusal names the pod and its violations. Every container and init container of those pods must set `readOnlyRootFilesystem: true` in its spec, and in the API, the web app and the worker, which keep running, a write to the root must fail.

## The enterprise edition

`edition.authEdition=enterprise` signs people in with your organisation's identity
provider over OpenID Connect (PRD-277). The chart then runs the enterprise web image
(`frontend.enterpriseImage`) and gives the API and the migration Job the settings below.
Rendering fails, naming the value and never what it holds, when a required one is missing
or is one the API would refuse at start.

| Value | API setting | |
|---|---|---|
| `edition.enterprise.publicUrl` | `FRONTEND_URL` | Required. The install's public address, `https://<host>`, shared by the web app and the API. The sign-in callback and the session cookie are on its origin. |
| `edition.enterprise.clientIpHeader` | `CLIENT_IP_HEADER` | Required. The header your ingress controller sets to the caller's address (`X-Real-IP` for ingress-nginx). Sign-in limits requests by it and the audit log records it. Name the header your proxy sets on every request, overwriting any value the caller sent, or the header your proxies append to, with `clientIpProxyCount`. It must hold one IP address or IP addresses separated by commas; rendering stops on `Forwarded`, whose values are `for=` parameters. |
| `edition.enterprise.clientIpProxyCount` | `CLIENT_IP_PROXY_COUNT` | How many of your own proxies append the caller's address to `clientIpHeader`: a plain whole number, 1 to 8. 1 for one load balancer, 2 for a content network in front of one. Count every proxy of yours that appends to the header, an ingress controller set to append among them. The API reads the value that many from the header's end. Required when `clientIpHeader` is `X-Forwarded-For` or `X-Original-Forwarded-For`; if your proxies append to any other header, state the count for it too. Empty: the address is the first value of the header's last line. Rendering stops on `0`, on anything outside 1 to 8 and on quoted text that is not a plain whole number (`"01"`, `"1.0"`). Write it unquoted without a leading zero: YAML reads `010` as 8 and `1.0` as 1, and the chart cannot tell them from 8 and 1. |
| `edition.enterprise.oidc.issuer` | `OIDC_ISSUER` | Required. The provider's issuer address, `https` only. |
| `edition.enterprise.oidc.clientId` | `OIDC_CLIENT_ID` | Required. The application's client id at the provider. |
| `edition.enterprise.oidc.displayName` | `OIDC_DISPLAY_NAME` | The sign-in button's text, 1 to 64 characters, no `<` or `>`. Empty: "your organisation". |
| `edition.enterprise.oidc.scopes` | `OIDC_SCOPES` | Separated by spaces, `openid` among them. Empty: `openid profile email`. |
| `edition.enterprise.oidc.requiredClaims` | `OIDC_REQUIRED_CLAIMS` | The access rule: `claim=value` rules separated by `;`. The API requires one for an issuer many organisations share, and checks the rules' syntax at start. |
| `edition.enterprise.oidc.endpointHosts` | `OIDC_ENDPOINT_HOSTS` | Host names, besides the issuer's, where the provider's metadata may put its token endpoint and keys. |
| `edition.enterprise.oidc.caConfigMap`, `caKey` | `OIDC_CA_BUNDLE_FILE` | A ConfigMap you create in the release's namespace holding the CA of a provider with a private certificate, under `caKey` (`ca.crt`). Mounted read-only at `/etc/automatos/oidc-ca` in the API and the Job. |
| `edition.enterprise.defaultRole` | `ENTERPRISE_DEFAULT_ROLE` | `none` (the default), `viewer`, `editor` or `admin`: the role in the default workspace for anyone the access rule lets in. Any role but `none` needs `requiredClaims`. |
| `edition.enterprise.bootstrapAdmins` | `ENTERPRISE_BOOTSTRAP_ADMINS` | `sub:<subject>` and `email:<address>` entries, separated by commas: install admins at their first sign-in. The API checks the entries at start. |
| `edition.defaultWorkspaceId` | `DEFAULT_WORKSPACE_ID` | The default workspace the Job makes, with no owner, that sign-ins are recorded in. |
| `edition.enterprise.hostedServices.themeRevocations` | `ENTERPRISE_CALLS_THEME_REVOCATIONS` | `false` (the default) or `true`: pull Automatos's theme revocation list from `THEME_REVOCATIONS_URL` (PRD-277 US-704). |
| `edition.enterprise.hostedServices.cloudSync` | `ENTERPRISE_CALLS_CLOUD_SYNC` | `false` (the default) or `true`: Sync with Automatos Cloud, its catalogue and the themes it lists, from `AUTOMATOS_CLOUD_URL`. Needs `themeRevocations`. |
| `edition.enterprise.hostedServices.cloudItems` | `ENTERPRISE_CALLS_CLOUD_ITEMS` | `false` (the default) or `true`: the agents, Playbooks, packages and plugins a Sync brings from Automatos Cloud. Needs `cloudSync`. |
| `edition.enterprise.devicePairing` | `ENTERPRISE_DEVICE_PAIRING` | `true` (the default) or `false`: whether people may pair their phones (PRD-277 US-601). Each person pairs their own phone from Settings, signed in as themselves, and the phone acts as that person with their current role. |

The hosted services are off by default: an enterprise install calls none of Automatos's hosted
services until you switch one on. Each is `true` or `false`, written without quotes; rendering
stops on any other value, on a name under `hostedServices` the chart does not know, on a service
switched on without the one it needs, and on any of the three settings in `config`. One never
switches on another. Switching one off stops its requests only: what the install already pulled
or brought (revoked themes and signing keys, Cloud themes and items) stays and still applies.
The chart gives `cloudSync` to the enterprise web container too, as `ENTERPRISE_CALLS_CLOUD_SYNC`:
browsers load Cloud screenshots from Automatos Cloud only while it is on, and the web app's content
security policy allows that one source only then (PRD-277 D-12).
What each one sends and brings back, and what an install gives up with it off, is in
[the deployment docs](../../docs/deployment-infrastructure/kubernetes-helm-chart.md#automatoss-hosted-services-in-the-enterprise-edition).

Device pairing is on by default. `devicePairing` is `true` or `false`, written without quotes;
rendering stops on any other value and on `ENTERPRISE_DEVICE_PAIRING` in `config`. With it off,
no phone pairs, every paired phone's token is refused, and the discovery document offers phones
sign-in only. The first time the install starts as an edition other than the one it last started
as, the API revokes every paired phone, so a phone paired under the local edition never carries
into the enterprise edition.

The client secret goes in the Secret as `OIDC_CLIENT_SECRET`, never in values. The
session clocks (`SESSION_IDLE_MINUTES`, `SESSION_ABSOLUTE_HOURS`), the claim names
(`OIDC_CLAIM_EMAIL`, `OIDC_CLAIM_NAME`, `OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES`) and
the pending sign-in limits (`OIDC_PENDING_SIGN_INS_MAX`, `OIDC_PENDING_SIGN_INS_PER_ADDRESS`)
have working defaults and go in `config` when you change them; the API checks them at start.
The session cookie's name and the CSRF header's are fixed in the enterprise web image, so
`config` may not change them. Nor may it set `FORWARDED_ALLOW_IPS`: the API reads the
caller's address from `clientIpHeader` and `clientIpProxyCount`, and leaves the server's own
forwarded-header handling trusting only its own host.

At the provider, register the sign-in callback `<publicUrl origin>/api/v1/auth/oidc/callback`
and the address after sign-out `<publicUrl origin>/sign-in`; the install notes print both.
Turn the ingress on (`ingress.enabled`, `ingress.host` the host of `publicUrl`, and `ingress.tls`):
the enterprise web app calls the API on its own origin, and the session cookie is a
`__Host-` cookie, sent over https only.

Example values:

```yaml
edition:
  authEdition: enterprise
  enterprise:
    publicUrl: https://automatos.example.com
    clientIpHeader: X-Real-IP
    bootstrapAdmins: email:admin@example.com
    oidc:
      issuer: https://id.example.com/realms/automatos
      clientId: automatos
ingress:
  enabled: true
  host: automatos.example.com
```

Behind an AWS application load balancer, which appends the connecting address to
`X-Forwarded-For`, name that header and say how many of your own proxies append to it:

```yaml
edition:
  enterprise:
    clientIpHeader: X-Forwarded-For
    clientIpProxyCount: 1   # 2 with a content network in front of the load balancer
```

The count is for proxies that append, the load balancer's default mode (`append`). One that
replaces the header with the address it saw leaves that one value: count it as 1, or name a
header it sets. One that passes the header on unchanged (`preserve`) adds nothing: leave it out of
the count. With the load balancer in `remove` mode the header never arrives: name another.
Behind ingress-nginx with its default settings, which sets `X-Forwarded-For` and `X-Real-IP` to the
address it saw and passes on the `X-Forwarded-For` it received as `X-Original-Forwarded-For`, name
`X-Original-Forwarded-For` and count the proxies in front of ingress-nginx.

## The worker in a namespace of its own

`worker.namespace` puts the worker in a namespace other than the release's, so it can have its own pod security level, network rules and quotas. A Secret, a ServiceAccount and a PersistentVolumeClaim belong to one namespace, and the chart never creates a namespace, so you provide, before `helm install`:

1. **The namespace**, with the pod security labels you want (`restricted` passes).
2. **The worker's Secret** in it, named in `worker.existingSecret`: `DATABASE_URL`, `REDIS_URL` and `WORKER_INTERNAL_TOKEN` (the same token as in `existingSecret`, which must now hold it too), and any of `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` and `OTEL_EXPORTER_OTLP_HEADERS` the worker uses.
3. **Two claims on one file system**: `workspaces.api.existingClaim` in the release's namespace, `workspaces.worker.existingClaim` in the worker's. A claim the chart made would be a disk of its own, so the chart makes none here: bind both to the same storage, for example two PersistentVolumes on one NFS export or file share, each with a `claimRef` to its claim. The storage must give uid 1000 or group 1000 write access (the worker writes, the API reads).
4. With `database.caConfigMap`: a ConfigMap of that name in the worker's namespace too.
5. With `serviceAccount.create=false`: the account `serviceAccount.worker.name` names, in the worker's namespace.

With `serviceAccount.create=true` the chart creates the worker's account in the worker's namespace. The API reaches the worker across the namespaces on port 8081: a network policy you add must allow it.

Rendering stops with a message on a combination that cannot work:

- `worker.namespace` that is not a valid namespace name;
- `worker.namespace` starting `kube-` (Kubernetes' own system namespaces, `kube-system` among them);
- `worker.namespace` set (and not the release's) without `worker.existingSecret`;
- `worker.namespace` set (and not the release's) without `workspaces.api.existingClaim` or without `workspaces.worker.existingClaim`.

## Upgrading to 0.11.0

- **The local edition gets one change: one extra ingress path.** With `ingress.enabled`, exactly `/.well-known/automatos.json`, the discovery document a phone reads, now goes to the API (`pathType: Exact`), between `/ws` and the web app's `/`. Before, the web app answered it. In the enterprise edition it answers an error until phone sign-in for that edition is built (PRD-277 Wave 6, US-602). The rest of `/.well-known/` stays the web app's, and `/exports/`, `/health`, `/metrics` and `/docs` are not routed. With the ingress off (the default) nothing changes but the chart's version label (`charts/automatos/tests/default_render_test.yaml` pins the default render).
- **The enterprise edition can be installed** (`edition.authEdition=enterprise`, PRD-277). It needs, before `helm install`:
  - `OIDC_CLIENT_SECRET` in the Secret `existingSecret` names: the API does not start without it;
  - `edition.enterprise.publicUrl`, `clientIpHeader`, `oidc.issuer` and `oidc.clientId`: rendering stops, naming the value, while one is missing or is one the API would refuse;
  - behind proxies that append to the client address header, `edition.enterprise.clientIpProxyCount`: how many of your own proxies append to it (1 behind one AWS application load balancer with `clientIpHeader: X-Forwarded-For`). Rendering stops while `clientIpHeader` is `X-Forwarded-For` or `X-Original-Forwarded-For` and the count is empty, while the count is not a plain whole number from 1 to 8, while a count is set with no `clientIpHeader`, while `clientIpHeader` is `Forwarded`, and while `config` sets `FORWARDED_ALLOW_IPS`;
  - the application registered at your identity provider with the two addresses the install notes print;
  - the ingress on, on the host of `publicUrl`, with TLS, so the web app and the API answer on one origin;
  - for a provider with a private certificate, a ConfigMap with its CA (`edition.enterprise.oidc.caConfigMap`);
  - to call any of Automatos's hosted services, each switched on under `edition.enterprise.hostedServices` (`themeRevocations`, `cloudSync`, `cloudItems`; all off by default, so nothing is called until you do).
  The chart then runs the enterprise web image (`frontend.enterpriseImage`), gives it `BACKEND_INTERNAL_URL`, `MEDIA_ORIGINS` (`frontend.mediaOrigins`) and `ENTERPRISE_CALLS_CLOUD_SYNC` (`edition.enterprise.hostedServices.cloudSync`), and gives the API and the migration Job the settings in [The enterprise edition](#the-enterprise-edition).
- **`edition.authEdition` must be exactly `local`, `enterprise` or `saas`**, in lower case with no spaces; rendering stops on any other spelling. 0.10.0 documented `local` alone and rendered any value.
- **`config` refuses `OIDC_CLIENT_SECRET` in every edition**: a secret is never a chart value. In the enterprise edition it also refuses each setting the chart sets (the three `ENTERPRISE_CALLS_*` hosted service settings among them) and `SESSION_COOKIE_NAME` and `SESSION_CSRF_HEADER`.
- **A released chart names one web image per published edition**, each by digest: `frontend.image` (local) and `frontend.enterpriseImage` (enterprise).

## Upgrading to 0.10.0

- **The worker runs as uid 1000 from its start, never as root** (PRD-280 US-201). Its pod sets `runAsNonRoot`, `runAsUser: 1000`, `runAsGroup: 1000`, `fsGroup: 1000` and `fsGroupChangePolicy: OnRootMismatch`, and its container adds no capability back (0.9.0 added `CHOWN`, `DAC_OVERRIDE`, `SETGID` and `SETUID`). It passes the "restricted" pod security level.
- **The worker re-owns nothing.** Before, it started as root and re-owned whatever on its volume was not uid 1000's. Now the storage must hand the volume over writable by the pod's group, as it already must for the API's data volume. With storage that honours `fsGroup` (most block-storage and CSI drivers) nothing changes: the kubelet gives the volume to group 1000, and what the 0.9.0 worker wrote is uid 1000's already. With storage that does not (some NFS-style drivers), the worker stops at start with `[entrypoint] uid 1000 cannot write /workspaces…`: give the volume's files to uid 1000 or group 1000 on the storage side, or use a storage class that honours `fsGroup`.
- **The chart needs a worker image from this release or later.** The image now starts as uid 1000, and its entrypoint knows it. An older worker image, started as uid 1000 by this chart, stops at its drop to the worker (`setpriv` cannot set groups without root). Upgrade the image with the chart: leave `worker.image.tag` and `worker.image.digest` empty, or pin a build from this release.
- **The worker image starts as uid 1000 everywhere, not only under this chart.** Anything else that starts it on a volume owned by root must now start it as root to keep the old behaviour: `user: "0:0"` in compose (the repository's compose file sets it), `docker run --user 0`, or on Railway `RAILWAY_RUN_UID=0` on the worker's service. Set that **before** an image from this release is deployed; set earlier it changes nothing, as the older image already starts as root. If it is missing, the worker prints `[entrypoint] uid 1000 cannot write /workspaces…` and exits 1 before it starts, touching no file.
- **The worker can have a namespace of its own** (PRD-280 US-202), so it can have its own pod security level and rules: set `worker.namespace`. Nothing changes unless you set it: with `worker.namespace`, `worker.existingSecret`, `workspaces.api.existingClaim` and `workspaces.worker.existingClaim` all empty, the chart renders what 0.9.0 did, but for the worker's start above and the chart's version label (`charts/automatos/tests/default_render_test.yaml` pins it). See "The worker in a namespace of its own" below for what you provide.
- **The worker can read a Secret of its own** (`worker.existingSecret`), in any namespace. It then reads no key from `existingSecret`, and `WORKER_INTERNAL_TOKEN` becomes a required key of both Secrets, the same value in each, so the two sides always share it: a pod whose Secret lacks it does not start (`CreateContainerConfigError`).
- **Each side of the agent-files volume can be given a claim that already exists** (`workspaces.api.existingClaim`, `workspaces.worker.existingClaim`). The chart then creates no `<fullname>-workspaces` claim; one it created before is kept (`helm.sh/resource-policy: keep`) and is yours to delete once nothing mounts it.
- **The web app, the worker and the migration Job take placement values, as the API does** (PRD-280 US-203): `nodeSelector`, `tolerations`, `affinity` and `topologySpreadConstraints` under `frontend`, `worker` and `migrations`, and `api.topologySpreadConstraints` beside the API's three. All empty by default: nothing is rendered, and pods are scheduled as before.

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
  and `worker`, and for the enterprise edition `frontend.enterpriseImage`). A `sha-<7>`
  tag is never re-pushed.
- **A digest:** `--set api.image.digest=sha256:<64 hex>`. The pod runs
  `repository@digest`; `crane digest` or `docker buildx imagetools inspect` prints
  a tag's digest.
- **At package time:** `helm package charts/automatos --app-version sha-1a2b3c4`
  stamps the chart's `appVersion`, the tag every component uses when its own tag
  and digest are empty.

## Tests

- **Unit tests, no cluster:** `helm unittest charts/automatos` (the
  [helm-unittest](https://github.com/helm-unittest/helm-unittest) plugin). CI runs
  them with `helm lint` and a kubeconform schema check against the oldest and the
  newest supported Kubernetes (`.github/workflows/chart.yml`).
- **An install under a strict namespace, in CI:** `chart.yml`'s `install` job builds
  the four images, installs the chart into a kind cluster with the values in
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
