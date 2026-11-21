{{/*
The enterprise edition (PRD-277): its web image and the settings the chart gives it.
*/}}

{{/*
The editions the chart documents (values.yaml), spelled exactly. The API trims and lower-cases
the value and reads an unknown one as saas, so a near miss would start an edition the chart did
not render for: the render stops instead, without printing the value.
*/}}
{{- define "automatos.validateEdition" -}}
{{- if not (has .Values.edition.authEdition (list "local" "enterprise" "saas")) }}
{{- fail "edition.authEdition must be exactly local, enterprise or saas, in lower case with no spaces: the chart chooses the web image and the sign-in settings by it, and the API reads it as its edition" }}
{{- end }}
{{- end }}

{{/* "true" when edition.authEdition is enterprise; stops the render on an unknown spelling. */}}
{{- define "automatos.enterprise" -}}
{{- include "automatos.validateEdition" . -}}
{{- if eq (toString .Values.edition.authEdition) "enterprise" -}}
true
{{- end -}}
{{- end }}

{{/*
"true" when one media origin is one the enterprise web server accepts, as
frontend/lib/enterprise/media-origins.ts checks it: https and a host (or `*.` over at least
two labels), or http on localhost or 127.0.0.1, with an optional port from 1 to 65535, and
nothing else. Called with the origin, its one trailing slash already dropped.
*/}}
{{- define "automatos.mediaOrigin" -}}
{{- $label := `[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?` -}}
{{- $host := printf `%s(?:\.%s)*` $label $label -}}
{{- $https := printf `(?i)^https://(?:\*\.%s\.%s|%s)(?::[0-9]{1,5})?$` $label $host $host -}}
{{- $local := `(?i)^http://(?:localhost|127\.0\.0\.1)(?::[0-9]{1,5})?$` -}}
{{- if or (regexMatch $https .) (regexMatch $local .) -}}
{{- $port := regexFind `:[0-9]{1,5}$` . | trimPrefix ":" -}}
{{- if or (not $port) (and (ge (atoi $port) 1) (le (atoi $port) 65535)) -}}
true
{{- end -}}
{{- end -}}
{{- end }}

{{/*
Stops the render on a frontend.mediaOrigins the enterprise web server would refuse at start
(PRD-279 US-402), or one set in an edition whose web image never reads it. The message names
the value, never what it holds.
*/}}
{{- define "automatos.validateMediaOrigins" -}}
{{- $raw := toString (.Values.frontend.mediaOrigins | default "") -}}
{{- if $raw }}
{{- if not (include "automatos.enterprise" .) }}
{{- fail (printf "frontend.mediaOrigins is set, but edition.authEdition is %q: only the enterprise web image reads its media origins when it starts, and the %s image has them fixed when it is built. Remove frontend.mediaOrigins" (toString .Values.edition.authEdition) (toString .Values.edition.authEdition)) }}
{{- end }}
{{- $tokens := regexSplit `[\s,]+` $raw -1 | compact }}
{{- $bad := or (gt (len $raw) 1024) (gt (len $tokens) 16) }}
{{- range $token := $tokens }}
{{- if not (include "automatos.mediaOrigin" (trimSuffix "/" $token)) }}
{{- $bad = true }}
{{- end }}
{{- end }}
{{- if $bad }}
{{- fail "frontend.mediaOrigins must list at most 16 origins, in at most 1024 characters, separated by commas or spaces, each https://<host>[:port] or https://*.<domain>.<suffix>[:port], or http://localhost[:port] or http://127.0.0.1[:port] for local development: the enterprise web server refuses to start on anything else (MEDIA_ORIGINS)" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
The web container's environment in the enterprise edition (PRD-277 US-505, PRD-279 US-402):
the media origins its server reads when it starts. BACKEND_INTERNAL_URL, which every edition
gets, is in frontend.yaml.
*/}}
{{- define "automatos.frontendEnterpriseEnv" -}}
{{- include "automatos.validateMediaOrigins" . -}}
{{- if include "automatos.enterprise" . }}
- name: MEDIA_ORIGINS
  value: {{ toString (.Values.frontend.mediaOrigins | default "") | quote }}
{{- end }}
{{- end }}
