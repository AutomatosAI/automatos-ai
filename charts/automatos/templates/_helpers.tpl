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

{{- define "automatos.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "automatos.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/* repository:tag, the tag defaulting to the chart's appVersion: (dict "ctx" $ "image" .Values.api.image). */}}
{{- define "automatos.image" -}}
{{- printf "%s:%s" .image.repository (default .ctx.Chart.AppVersion .image.tag) }}
{{- end }}

{{- define "automatos.secretName" -}}
{{- required "existingSecret is required: the Secret holding DATABASE_URL, REDIS_URL, API_KEY and CREDENTIAL_ENCRYPTION_KEY (see values.yaml)" .Values.existingSecret }}
{{- end }}

{{- define "automatos.workspacesClaim" -}}
{{- printf "%s-workspaces" (include "automatos.fullname" .) }}
{{- end }}

{{/*
The API's environment, shared by the API pods and the migration Job. Rendered
inline rather than through a ConfigMap: the Job is a pre-install hook, so it runs
before any of the release's ordinary resources exist.
*/}}
{{- define "automatos.apiEnv" -}}
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
  value: {{ printf "http://%s-worker:8081" (include "automatos.fullname" .) | quote }}
{{- end }}
{{- range $name, $value := .Values.config }}
- name: {{ $name }}
  value: {{ $value | toString | quote }}
{{- end }}
{{- range $key := list "DATABASE_URL" "REDIS_URL" "API_KEY" "CREDENTIAL_ENCRYPTION_KEY" }}
- name: {{ $key }}
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      key: {{ $key }}
{{- end }}
- name: WORKER_INTERNAL_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      key: WORKER_INTERNAL_TOKEN
      optional: true
{{- end }}
