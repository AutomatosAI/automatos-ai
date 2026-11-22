{{/* Names and labels. */}}
{{- define "automatos.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "automatos.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else if contains (include "automatos.name" .) .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "automatos.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "automatos.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "automatos.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Selector labels for one component: (dict "ctx" $ "component" "api"). */}}
{{- define "automatos.selectorLabels" -}}
app.kubernetes.io/name: {{ include "automatos.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
The components that run under an account of their own (PRD-280 US-101), as the keys of
serviceAccount in values.yaml: (include "automatos.serviceAccountComponents" . | splitList " ").
Only those that are on, in the order serviceaccount.yaml renders them.
*/}}
{{- define "automatos.serviceAccountComponents" -}}
api{{ if .Values.frontend.enabled }} frontend{{ end }}{{ if .Values.worker.enabled }} worker{{ end }} migrate
{{- end }}

{{/*
Stops the render on serviceAccount values that no longer do anything. Until chart 0.7.0 one
account served every pod; a value meant for it would otherwise be dropped without a word.
*/}}
{{- define "automatos.validateServiceAccounts" -}}
{{- $accounts := .Values.serviceAccount | default dict -}}
{{- if $accounts.name }}
{{- fail "serviceAccount.name: since chart 0.7.0 each component has its own account. Set serviceAccount.api.name, serviceAccount.frontend.name, serviceAccount.worker.name and serviceAccount.migrate.name instead (with serviceAccount.create=false they may all name the same account), and remove serviceAccount.name" }}
{{- end }}
{{- if $accounts.annotations }}
{{- fail "serviceAccount.annotations: since chart 0.7.0 each component has its own account. Put the annotations on the account that needs them, e.g. serviceAccount.api.annotations, and remove serviceAccount.annotations" }}
{{- end }}
{{- range $key, $_ := $accounts }}
{{- if not (has $key (list "create" "name" "annotations" "api" "frontend" "worker" "migrate")) }}
{{- fail (printf "serviceAccount.%s: unknown key. The accounts are serviceAccount.api, serviceAccount.frontend (the web app), serviceAccount.worker and serviceAccount.migrate" $key) }}
{{- end }}
{{- end }}
{{- if $accounts.create }}
{{- $seen := dict }}
{{- range $component := include "automatos.serviceAccountComponents" $ | splitList " " }}
{{- $name := include "automatos.serviceAccountNameUnchecked" (dict "ctx" $ "component" $component) }}
{{- with get $seen $name }}
{{- fail (printf "serviceAccount.%s.name and serviceAccount.%s.name are both %q: with serviceAccount.create=true each component's account needs a name of its own" . $component $name) }}
{{- end }}
{{- $_ := set $seen $name $component }}
{{- end }}
{{- end }}
{{- end }}

{{/*
A component's account, without the checks: (dict "ctx" $ "component" "api"). With
serviceAccount.create, the chart's own account, <fullname>-<component> unless
serviceAccount.<component>.name renames it. Without, the account that value names, or the
namespace's default account when it names none.
*/}}
{{- define "automatos.serviceAccountNameUnchecked" -}}
{{- $account := index .ctx.Values.serviceAccount .component | default dict -}}
{{- if .ctx.Values.serviceAccount.create -}}
{{- default (printf "%s-%s" (include "automatos.fullname" .ctx) .component) $account.name -}}
{{- else -}}
{{- default "default" $account.name -}}
{{- end -}}
{{- end }}

{{/* A component's account, after the checks above: (dict "ctx" $ "component" "api"). */}}
{{- define "automatos.serviceAccountName" -}}
{{- include "automatos.validateServiceAccounts" .ctx -}}
{{- include "automatos.serviceAccountNameUnchecked" . -}}
{{- end }}

{{/*
A component's image: (dict "ctx" $ "image" .Values.api.image). repository@digest when
image.digest is set, so the node runs exactly that build; otherwise repository:tag,
the tag defaulting to the chart's appVersion.
*/}}
{{- define "automatos.image" -}}
{{- with .image.digest }}
{{- if not (regexMatch `^sha256:[0-9a-f]{64}$` .) }}
{{- fail (printf "%s: image.digest must be sha256:<64 hex digits>, got %q" $.image.repository .) }}
{{- end }}
{{- end }}
{{- if .image.digest }}
{{- printf "%s@%s" .image.repository .image.digest }}
{{- else }}
{{- printf "%s:%s" .image.repository (toString (default .ctx.Chart.AppVersion .image.tag)) }}
{{- end }}
{{- end }}

{{/*
A component's imagePullPolicy: (dict "ctx" $ "image" .Values.api.image). image.pullPolicy
when set. Otherwise Always for a tag that moves (edge, or a MAJOR.MINOR release line),
so a node never keeps an older build under it, and IfNotPresent for a digest or a tag
that never moves (sha-<7>, tree-<12>, weekly-<date>, a full version, or an appVersion
stamped at package time).
*/}}
{{- define "automatos.imagePullPolicy" -}}
{{- $tag := toString (default .ctx.Chart.AppVersion .image.tag) -}}
{{- if .image.pullPolicy -}}
{{ .image.pullPolicy }}
{{- else if .image.digest -}}
IfNotPresent
{{- else if or (eq $tag "edge") (regexMatch `^[0-9]+\.[0-9]+$` $tag) -}}
Always
{{- else -}}
IfNotPresent
{{- end -}}
{{- end }}

{{- define "automatos.secretName" -}}
{{- required "existingSecret is required: the Secret holding DATABASE_URL, REDIS_URL, API_KEY and CREDENTIAL_ENCRYPTION_KEY (see values.yaml)" .Values.existingSecret }}
{{- end }}

{{/*
The worker in a namespace of its own (PRD-280 US-202). "true" when the worker is on and
worker.namespace names a namespace other than the release's. The chart never creates that
namespace or labels it: the operator does (PRD-280 D-5).
*/}}
{{- define "automatos.workerApart" -}}
{{- if and .Values.worker.enabled .Values.worker.namespace (ne (toString .Values.worker.namespace) .Release.Namespace) -}}
true
{{- end -}}
{{- end }}

{{/* The worker's Secret: worker.existingSecret, or the chart's existingSecret. */}}
{{- define "automatos.workerSecretName" -}}
{{- default (include "automatos.secretName" .) .Values.worker.existingSecret -}}
{{- end }}

{{/* The address the API reaches the worker at: its Service, by namespace when it has its own. */}}
{{- define "automatos.workerUrl" -}}
{{- if include "automatos.workerApart" . -}}
{{- printf "http://%s-worker.%s.svc:8081" (include "automatos.fullname" .) .Values.worker.namespace -}}
{{- else -}}
{{- printf "http://%s-worker:8081" (include "automatos.fullname" .) -}}
{{- end -}}
{{- end }}

{{/*
The agent-files volume's claims, one for each side: (dict "ctx" $ "side" "api") or "worker".
A side's workspaces.<side>.existingClaim when set. In one namespace, the other side's claim
when only that one is named, so one claim serves both; otherwise the chart's own claim.
*/}}
{{- define "automatos.workspacesExistingClaim" -}}
{{- (index (.ctx.Values.workspaces | default dict) .side | default dict).existingClaim | default "" -}}
{{- end }}

{{- define "automatos.workspacesSideClaim" -}}
{{- $own := include "automatos.workspacesExistingClaim" . -}}
{{- $other := include "automatos.workspacesExistingClaim" (dict "ctx" .ctx "side" (ternary "worker" "api" (eq .side "api"))) -}}
{{- if $own -}}
{{- $own -}}
{{- else if and $other (not (include "automatos.workerApart" .ctx)) -}}
{{- $other -}}
{{- else -}}
{{- printf "%s-workspaces" (include "automatos.fullname" .ctx) -}}
{{- end -}}
{{- end }}

{{/* The API's side of the agent-files volume. */}}
{{- define "automatos.workspacesClaim" -}}
{{- include "automatos.workspacesSideClaim" (dict "ctx" . "side" "api") -}}
{{- end }}

{{/* The worker's side of the agent-files volume. */}}
{{- define "automatos.workerWorkspacesClaim" -}}
{{- include "automatos.workspacesSideClaim" (dict "ctx" . "side" "worker") -}}
{{- end }}

{{/* "true" when the chart creates the agent-files claim: neither side names one of its own. */}}
{{- define "automatos.workspacesClaimCreated" -}}
{{- if not (or (include "automatos.workspacesExistingClaim" (dict "ctx" . "side" "api")) (include "automatos.workspacesExistingClaim" (dict "ctx" . "side" "worker"))) -}}
true
{{- end -}}
{{- end }}

{{/*
Stops the render on a worker namespace that cannot work. A Secret, a ServiceAccount and a
PersistentVolumeClaim belong to one namespace: the worker apart needs a Secret named for it
there, and each side of the agent-files volume a claim the operator made and bound to one
file system, since a claim the chart made would be a disk of its own.
*/}}
{{- define "automatos.validateWorker" -}}
{{- with .Values.worker.namespace }}
{{- if not (regexMatch `^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$` (toString .)) }}
{{- fail (printf "worker.namespace must be a namespace name (lower-case letters, digits and '-', at most 63, starting and ending with a letter or digit), got %q" (toString .)) }}
{{- end }}
{{- if hasPrefix "kube-" (toString .) }}
{{- fail (printf "worker.namespace is %q: a name starting kube- is reserved for Kubernetes' own system namespaces. Create a namespace of your own for the worker and name it here" (toString .)) }}
{{- end }}
{{- end }}
{{- if include "automatos.workerApart" . }}
{{- if not .Values.worker.existingSecret }}
{{- fail (printf "worker.namespace is %q, so the worker needs a Secret of its own in that namespace: a Secret cannot be read from another one. Create it there with DATABASE_URL, REDIS_URL and WORKER_INTERNAL_TOKEN (the same token as the Secret existingSecret names), and any of ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN and OTEL_EXPORTER_OTLP_HEADERS the worker uses, then name it in worker.existingSecret" .Values.worker.namespace) }}
{{- end }}
{{- range $side := list "api" "worker" }}
{{- if not (include "automatos.workspacesExistingClaim" (dict "ctx" $ "side" $side)) }}
{{- fail (printf "worker.namespace is %q, so workspaces.%s.existingClaim must name a claim that already exists: a claim belongs to one namespace, so the chart cannot make one the API (in %q, read-only) and the worker (in %q) both mount. Create workspaces.api.existingClaim in the release's namespace and workspaces.worker.existingClaim in the worker's, bound to one file system" $.Values.worker.namespace $side $.Release.Namespace $.Values.worker.namespace) }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Where a component's pods may run (PRD-280 US-203): its nodeSelector, tolerations, affinity
and topologySpreadConstraints, each rendered as given, at the pod spec's indent; nothing for
an empty one. Called with the component's values: (include "automatos.placement" .Values.worker).
*/}}
{{- define "automatos.placement" -}}
{{- with .nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .topologySpreadConstraints }}
topologySpreadConstraints:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{- define "automatos.apiDataClaim" -}}
{{- printf "%s-api-data" (include "automatos.fullname" .) }}
{{- end }}

{{/*
"true" when the API's data volume can attach to only one node (or pod) at a time.
*/}}
{{- define "automatos.apiDataSingleAttach" -}}
{{- if and .Values.api.data.enabled (or (has "ReadWriteOnce" .Values.api.data.accessModes) (has "ReadWriteOncePod" .Values.api.data.accessModes)) -}}
true
{{- end -}}
{{- end }}

{{/*
The API's scratch volumes under a read-only root, space-separated; api-deployment.yaml
mounts each of them (/app/documents only with api.data.enabled=false, the one case the
document paths are checked against this list).
*/}}
{{- define "automatos.apiScratchDirs" -}}
/tmp /home/automatos /app/logs /app/documents
{{- end }}

{{/*
"true" when a path the API writes (a config value; a relative one is under its working
directory, /app) is under one of its scratch volumes once cleaned of "." and "..".
*/}}
{{- define "automatos.apiScratchPath" -}}
{{- $path := toString . -}}
{{- if not (isAbs $path) }}{{ $path = printf "/app/%s" $path }}{{ end -}}
{{- $path = clean $path -}}
{{- range $dir := include "automatos.apiScratchDirs" $ | splitList " " -}}
{{- if or (eq $path $dir) (hasPrefix (printf "%s/" $dir) $path) }}true{{ end -}}
{{- end -}}
{{- end }}

{{/*
Stops a release at render time, before any hook runs, on values that can't work. A name
in `config` that the chart also sets would be rendered twice, and an upgrade then fails
partway through. A second replica on a single-attach volume would hang waiting for it. A
document path of the operator's own outside the API's scratch volumes, under a read-only
root, would fail at the first upload.
*/}}
{{- define "automatos.validateApi" -}}
{{- include "automatos.validateEdition" . -}}
{{- include "automatos.validateLogFormat" . -}}
{{- include "automatos.validateLogLevel" . -}}
{{- $config := .Values.config | default dict -}}
{{- range $key := list "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" }}
{{- if hasKey $config $key }}
{{- fail (printf "config.%s: move it to the Secret named by existingSecret, under the same key, so it isn't plain text in the pod spec" $key) }}
{{- end }}
{{- end }}
{{- if hasKey $config "OIDC_CLIENT_SECRET" }}
{{- fail "config.OIDC_CLIENT_SECRET: a secret is never a chart value. Put it in the Secret existingSecret names, under the key OIDC_CLIENT_SECRET, and remove it from config" }}
{{- end }}
{{- if .Values.api.data.enabled }}
{{- range $key := list "DOCUMENT_UPLOAD_DIR" "DOCUMENT_STORAGE_DIR" }}
{{- if hasKey $config $key }}
{{- fail (printf "config.%s: api.data sets this. Remove it from config, or set api.data.enabled=false to manage the API's files yourself" $key) }}
{{- end }}
{{- end }}
{{- end }}
{{- if .Values.api.readOnlyRootFilesystem }}
{{- if hasKey $config "HOME" }}
{{- fail "config.HOME: with a read-only root (api.readOnlyRootFilesystem) the chart sets HOME to the API's home scratch volume, /home/automatos. Remove it from config, or set api.readOnlyRootFilesystem=false" }}
{{- end }}
{{- if not .Values.api.data.enabled }}
{{- range $key := list "DOCUMENT_UPLOAD_DIR" "DOCUMENT_STORAGE_DIR" }}
{{- if and (hasKey $config $key) (not (include "automatos.apiScratchPath" (get $config $key))) }}
{{- fail (printf "config.%s is %q: with api.data.enabled=false and a read-only root (api.readOnlyRootFilesystem) the API can write only under its scratch volumes, %s. Use a path under one of them, or set api.readOnlyRootFilesystem=false" $key (get $config $key | toString) (include "automatos.apiScratchDirs" $ | splitList " " | join ", ")) }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
{{- if and (include "automatos.apiDataSingleAttach" .) (gt (int .Values.api.replicas) 1) }}
{{- fail "api.replicas above 1 needs api.data.accessModes without ReadWriteOnce or ReadWriteOncePod. A claim's access modes can't change after it's created, so set this at install" }}
{{- end }}
{{- end }}

{{/*
TLS to PostgreSQL (PRD-278 W3), for the API, the migration Job and the worker: the same
two settings the app reads, and the CA file mounted from `database.caConfigMap`. Nothing
when `database` is left empty, outside the enterprise edition. Stops rendering on values
the app would refuse at start: an unknown mode, a CA without a mode that checks it, a mode
that checks with no CA, and, in the enterprise edition, any mode but verify-full (D-6).
*/}}
{{- define "automatos.databaseCaDir" -}}
/etc/automatos/database-ca
{{- end }}

{{- define "automatos.validateDatabaseTls" -}}
{{- $config := .Values.config | default dict -}}
{{- range $key := list "DATABASE_SSLMODE" "DATABASE_SSLROOTCERT" }}
{{- if hasKey $config $key }}
{{- fail (printf "config.%s: set database.sslmode and database.caConfigMap instead, so the worker gets it too" $key) }}
{{- end }}
{{- end }}
{{- with .Values.database.sslmode }}
{{- if not (has . (list "disable" "allow" "prefer" "require" "verify-ca" "verify-full")) }}
{{- fail (printf "database.sslmode must be disable, allow, prefer, require, verify-ca or verify-full, got %q" .) }}
{{- end }}
{{- end }}
{{- $checking := has .Values.database.sslmode (list "verify-ca" "verify-full") }}
{{- if and .Values.database.caConfigMap (not $checking) }}
{{- fail "database.caConfigMap needs database.sslmode verify-ca or verify-full: the CA file is used only to check the server's certificate" }}
{{- end }}
{{- if and $checking (not .Values.database.caConfigMap) }}
{{- fail "database.sslmode verify-ca and verify-full need database.caConfigMap: a ConfigMap holding the CA file that signed the database server's certificate" }}
{{- end }}
{{- if and (include "automatos.enterprise" .) (ne (.Values.database.sslmode | default "") "verify-full") }}
{{- fail "database.sslmode must be verify-full in the enterprise edition, with database.caConfigMap: the API connects to its database only after checking the server's certificate and host name (PRD-278 D-6)" }}
{{- end }}
{{- end }}

{{- define "automatos.databaseTlsEnv" -}}
{{- include "automatos.validateDatabaseTls" . -}}
{{- with .Values.database.sslmode }}
- name: DATABASE_SSLMODE
  value: {{ . | quote }}
{{- end }}
{{- if .Values.database.caConfigMap }}
- name: DATABASE_SSLROOTCERT
  value: {{ printf "%s/%s" (include "automatos.databaseCaDir" .) .Values.database.caKey | quote }}
{{- end }}
{{- end }}

{{/* The CA file's volume and its read-only mount. Nothing without database.caConfigMap. */}}
{{- define "automatos.databaseCaVolume" -}}
{{- with .Values.database.caConfigMap }}
- name: database-ca
  configMap:
    name: {{ . }}
{{- end }}
{{- end }}

{{- define "automatos.databaseCaMount" -}}
{{- if .Values.database.caConfigMap }}
- name: database-ca
  mountPath: {{ include "automatos.databaseCaDir" . }}
  readOnly: true
{{- end }}
{{- end }}

{{/*
The API's environment, shared by the API pods and the migration Job. Rendered
inline rather than through a ConfigMap: the Job is a pre-install hook, so it runs
before any of the release's ordinary resources exist.
*/}}
{{- define "automatos.apiEnv" -}}
{{- include "automatos.validateApi" . -}}
{{- $secret := include "automatos.secretName" . -}}
- name: AUTH_EDITION
  value: {{ .Values.edition.authEdition | quote }}
{{- if eq .Values.edition.authEdition "local" }}
- name: DEFAULT_WORKSPACE_ID
  value: {{ required "edition.defaultWorkspaceId is required for the local edition" .Values.edition.defaultWorkspaceId | quote }}
- name: PLATFORM_KEY_WORKSPACE_ID
  value: {{ .Values.edition.defaultWorkspaceId | quote }}
- name: LOCAL_OPERATOR_EMAIL
  value: {{ .Values.edition.localOperatorEmail | quote }}
{{- end }}
{{- include "automatos.enterpriseApiEnv" . }}
- name: WORKSPACE_VOLUME_PATH
  value: /workspaces
{{- if .Values.sessionMode.enabled }}
{{- if ne .Values.edition.authEdition "local" }}
{{- fail "sessionMode.enabled needs edition.authEdition=local: session mode runs the operator's own CLI sessions, and the hosted edition refuses it" }}
{{- end }}
- name: CLI_RUNTIME_ENABLED
  value: "true"
- name: CLI_SESSION_FILE_UPLOAD
  value: "true"
- name: CLI_SESSION_UPLOAD_MAX_TOTAL_MB
  value: {{ .Values.sessionMode.uploadMaxTotalMb | toString | quote }}
{{- end }}
{{- if .Values.worker.enabled }}
- name: WORKER_INTERNAL_URL
  value: {{ include "automatos.workerUrl" . | quote }}
{{- end }}
{{- include "automatos.databaseTlsEnv" . }}
{{- range $name, $value := .Values.config }}
- name: {{ $name }}
  value: {{ $value | toString | quote }}
{{- end }}
{{- $required := list "DATABASE_URL" "REDIS_URL" "API_KEY" "CREDENTIAL_ENCRYPTION_KEY" }}
{{- $optional := list "WORKER_INTERNAL_TOKEN" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" }}
{{- if and .Values.worker.enabled .Values.worker.existingSecret }}
{{- /* The worker's own Secret must hold the token (PRD-280 US-202), so the API's must too. */}}
{{- $required = append $required "WORKER_INTERNAL_TOKEN" }}
{{- $optional = without $optional "WORKER_INTERNAL_TOKEN" }}
{{- end }}
{{- if include "automatos.enterprise" . }}
{{- /* The identity provider's client secret (PRD-277 US-701): from the Secret, never a value. */}}
{{- $required = append $required "OIDC_CLIENT_SECRET" }}
{{- end }}
{{- range $key := $required }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      key: {{ $key }}
{{- end }}
{{- range $key := $optional }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      key: {{ $key }}
      optional: true
{{- end }}
{{- end }}

{{/*
OpenTelemetry (PRD-256) for one component: (dict "ctx" $ "service" "automatos-api"), and
"secret" for a component that reads a Secret of its own (the worker, PRD-280 US-202).
Nothing when otel.enabled is false.
*/}}
{{- define "automatos.otelEnv" -}}
{{- $ctx := .ctx -}}
{{- with $ctx.Values.otel }}
{{- if .enabled }}
- name: OTEL_ENABLED
  value: "true"
- name: OTEL_SERVICE_NAME
  value: {{ $.service | quote }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ required "otel.endpoint is required when otel.enabled is true: the OTLP/HTTP address of your collector" .endpoint | quote }}
- name: OTEL_TRACES_SAMPLER_RATIO
  value: {{ .samplerRatio | toString | quote }}
- name: OTEL_EXPORTER_OTLP_HEADERS
  valueFrom:
    secretKeyRef:
      name: {{ $.secret | default (include "automatos.secretName" $ctx) }}
      key: OTEL_EXPORTER_OTLP_HEADERS
      optional: true
- name: OTEL_POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: OTEL_POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
{{- $extra := "" }}
{{- if .resourceAttributes }}
{{- $extra = printf ",%s" .resourceAttributes }}
{{- end }}
- name: OTEL_RESOURCE_ATTRIBUTES
  value: {{ printf "k8s.pod.name=$(OTEL_POD_NAME),k8s.namespace.name=$(OTEL_POD_NAMESPACE)%s" $extra | quote }}
{{- end }}
{{- end }}
{{- end }}
