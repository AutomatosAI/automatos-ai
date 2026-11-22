{{/*
The logs (PRD-282 US-303). config.LOG_FORMAT, like config.LOG_LEVEL, reaches the API and
the migration Job with the rest of `config` (automatos.apiEnv); the worker, which reads no other
`config` value, gets LOG_FORMAT from it too. The migration tool's own output stays text whatever
the format. Unset, nothing is added anywhere, and every pod writes text as before.

At run time an unknown value of either is the default (text, INFO) with one warning; rendering
stops on one here instead, so a misspelling is seen before anything is deployed.
*/}}

{{/* The formats the API and the worker read, after trimming and in any case; empty is text.
A test keeps this list and log_settings.LOG_FORMATS the same. */}}
{{- define "automatos.logFormats" -}}
text json
{{- end }}

{{/* The level names the API and the worker read, after trimming and in any case; empty is INFO.
A test keeps this list and log_settings.LOG_LEVELS the same. */}}
{{- define "automatos.logLevels" -}}
DEBUG INFO WARNING WARN ERROR CRITICAL FATAL
{{- end }}

{{/*
Stops the render on a LOG_FORMAT the API and the worker cannot use, naming the value's key and
never what it holds. Included by automatos.validateApi and the worker's template.
*/}}
{{- define "automatos.validateLogFormat" -}}
{{- $config := .Values.config | default dict -}}
{{- if hasKey $config "LOG_FORMAT" }}
{{- $format := get $config "LOG_FORMAT" | toString | trim | lower -}}
{{- if and $format (not (has $format (include "automatos.logFormats" . | splitList " "))) }}
{{- fail "config.LOG_FORMAT must be text or json, or unset for text" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Stops the render on a LOG_LEVEL the API cannot use, naming the value's key and never what it
holds. Included by automatos.validateApi.
*/}}
{{- define "automatos.validateLogLevel" -}}
{{- $config := .Values.config | default dict -}}
{{- if hasKey $config "LOG_LEVEL" }}
{{- $level := get $config "LOG_LEVEL" | toString | trim | upper -}}
{{- if and $level (not (has $level (include "automatos.logLevels" . | splitList " "))) }}
{{- fail "config.LOG_LEVEL must be DEBUG, INFO, WARNING, WARN, ERROR, CRITICAL or FATAL, or unset for INFO" }}
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
