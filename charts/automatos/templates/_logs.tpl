{{/*
The log format (PRD-282 US-303). config.LOG_FORMAT, like config.LOG_LEVEL, reaches the API and
the migration Job with the rest of `config` (automatos.apiEnv); the worker, which reads no other
`config` value, gets LOG_FORMAT from it too. Unset, nothing is added anywhere, and every pod
writes text as before.
*/}}

{{/* The formats the API and the worker read, after trimming and in any case; empty is text. */}}
{{- define "automatos.logFormats" -}}
text json
{{- end }}

{{/*
Stops the render on a LOG_FORMAT the API and the worker would refuse at start, naming the
value's key and never what it holds. Included by automatos.validateApi and the worker's template.
*/}}
{{- define "automatos.validateLogFormat" -}}
{{- $config := .Values.config | default dict -}}
{{- if hasKey $config "LOG_FORMAT" }}
{{- $format := get $config "LOG_FORMAT" | toString | trim | lower -}}
{{- if and $format (not (has $format (include "automatos.logFormats" . | splitList " "))) }}
{{- fail "config.LOG_FORMAT must be text or json, or unset for text: the API and the worker refuse to start on any other value" }}
{{- end }}
{{- end }}
{{- end }}

{{/* The worker's LOG_FORMAT, from config.LOG_FORMAT; nothing when it is unset. */}}
{{- define "automatos.workerLogFormatEnv" -}}
{{- $config := .Values.config | default dict -}}
{{- if hasKey $config "LOG_FORMAT" }}
- name: LOG_FORMAT
  value: {{ get $config "LOG_FORMAT" | toString | quote }}
{{- end }}
{{- end }}
