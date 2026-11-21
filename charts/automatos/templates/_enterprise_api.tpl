{{/*
The enterprise edition's API settings (PRD-277 US-701), shared by the API and the migration
Job through automatos.apiEnv. Each value under edition.enterprise sets one API setting; the
client secret comes from the Secret only. Rendering stops on what the API would refuse at start
(core/edition_start_up.py), where the chart can know it, naming the value and never what it
holds. What only the API can judge (the access rule's and the first admins' syntax, an issuer
many organisations share, whether the CA file holds a certificate) it judges at start.
*/}}

{{/* Where the identity provider's certificate authority is mounted. */}}
{{- define "automatos.oidcCaDir" -}}
/etc/automatos/oidc-ca
{{- end }}

{{/* The settings the chart sets in the enterprise edition, which `config` must not set too. */}}
{{- define "automatos.enterpriseSettings" -}}
DEFAULT_WORKSPACE_ID FRONTEND_URL CLIENT_IP_HEADER ENTERPRISE_BOOTSTRAP_ADMINS ENTERPRISE_DEFAULT_ROLE OIDC_ISSUER OIDC_CLIENT_ID OIDC_DISPLAY_NAME OIDC_SCOPES OIDC_REQUIRED_CLAIMS OIDC_ENDPOINT_HOSTS OIDC_CA_BUNDLE_FILE
{{- end }}

{{/*
"true" when an address is https with a host and, if it has one, a port of at most 65535, as
the API's urlsplit check reads it (OIDC_ISSUER, FRONTEND_URL). Called with the value.
*/}}
{{- define "automatos.httpsAddress" -}}
{{- $authority := regexFind `(?i)^https://[^/?#]*` (toString . | trim) -}}
{{- if $authority -}}
{{- $hostPort := regexReplaceAll `^[^@]*@` (substr 8 (len $authority) $authority) "" -}}
{{- if regexMatch `^(\[[0-9A-Fa-f:.]+\]|[^:\[\]]+)(:[0-9]*)?$` $hostPort -}}
{{- $port := regexFind `:[0-9]*$` (regexReplaceAll `^\[[^\]]*\]` $hostPort "") | trimPrefix ":" -}}
{{- if or (not $port) (le (atoi $port) 65535) -}}
true
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}

{{/* "true" when a UUID's text is one Python's UUID() reads (DEFAULT_WORKSPACE_ID). */}}
{{- define "automatos.uuid" -}}
{{- $hex := toString . | trim | lower | replace "urn:" "" | replace "uuid:" "" | trimAll "{}" | replace "-" "" -}}
{{- if regexMatch `^[0-9a-f]{32}$` $hex -}}
true
{{- end -}}
{{- end }}

{{/* Stops the render on a `config` entry the enterprise edition sets elsewhere. */}}
{{- define "automatos.validateEnterpriseConfig" -}}
{{- $config := .Values.config | default dict -}}
{{- if hasKey $config "OIDC_CLIENT_SECRET" }}
{{- fail "config.OIDC_CLIENT_SECRET: a secret is never a chart value. Put it in the Secret existingSecret names, under the key OIDC_CLIENT_SECRET, and remove it from config" }}
{{- end }}
{{- range $key := include "automatos.enterpriseSettings" $ | splitList " " }}
{{- if hasKey $config $key }}
{{- fail (printf "config.%s: in the enterprise edition the chart sets it from edition.enterprise (DEFAULT_WORKSPACE_ID from edition.defaultWorkspaceId). Set it there, and remove it from config" $key) }}
{{- end }}
{{- end }}
{{- range $key := list "SESSION_COOKIE_NAME" "SESSION_CSRF_HEADER" }}
{{- if hasKey $config $key }}
{{- fail (printf "config.%s: the enterprise web image sends the session cookie __Host-automatos_session and the header X-Automatos-CSRF, fixed when it is built, so the API must keep them. Remove it from config" $key) }}
{{- end }}
{{- end }}
{{- end }}

{{/* Stops the render on a required enterprise value that is missing, in a fixed order. */}}
{{- define "automatos.validateEnterpriseRequired" -}}
{{- $enterprise := .Values.edition.enterprise | default dict -}}
{{- $oidc := $enterprise.oidc | default dict -}}
{{- if not (toString ($oidc.issuer | default "") | trim) }}
{{- fail "edition.enterprise.oidc.issuer is required in the enterprise edition: the identity provider's issuer address, https only (OIDC_ISSUER)" }}
{{- end }}
{{- if not (toString ($oidc.clientId | default "") | trim) }}
{{- fail "edition.enterprise.oidc.clientId is required in the enterprise edition: the application's client id at the identity provider (OIDC_CLIENT_ID)" }}
{{- end }}
{{- if not (toString ($enterprise.publicUrl | default "") | trim) }}
{{- fail "edition.enterprise.publicUrl is required in the enterprise edition: the install's public address, https://<host>, which the web app and the API share (FRONTEND_URL)" }}
{{- end }}
{{- if not (toString ($enterprise.clientIpHeader | default "") | trim) }}
{{- fail "edition.enterprise.clientIpHeader is required in the enterprise edition: the header your ingress controller or proxy sets to the caller's address, which sign-in limits requests by and writes to the audit log (CLIENT_IP_HEADER)" }}
{{- end }}
{{- if not (include "automatos.uuid" (.Values.edition.defaultWorkspaceId | default "")) }}
{{- fail "edition.defaultWorkspaceId must be a UUID in the enterprise edition: the default workspace sign-ins are recorded in, which the migration Job makes (DEFAULT_WORKSPACE_ID)" }}
{{- end }}
{{- end }}

{{/* Stops the render on an enterprise value the API refuses at start. */}}
{{- define "automatos.validateEnterpriseValues" -}}
{{- $enterprise := .Values.edition.enterprise | default dict -}}
{{- $oidc := $enterprise.oidc | default dict -}}
{{- if not (include "automatos.httpsAddress" $oidc.issuer) }}
{{- fail "edition.enterprise.oidc.issuer must be an https address with a host (OIDC_ISSUER): the API refuses to start otherwise" }}
{{- end }}
{{- if not (include "automatos.httpsAddress" $enterprise.publicUrl) }}
{{- fail "edition.enterprise.publicUrl must be an https address with a host (FRONTEND_URL): the API refuses to start otherwise" }}
{{- end }}
{{- if not (regexMatch "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$" (toString $enterprise.clientIpHeader | trim)) }}
{{- fail "edition.enterprise.clientIpHeader must be one header name, letters, digits and - (CLIENT_IP_HEADER): the API refuses to start otherwise" }}
{{- end }}
{{- include "automatos.validateEnterpriseRole" . }}
{{- include "automatos.validateEnterpriseOidcText" . }}
{{- end }}

{{/* The default role: one the API knows, and with an access rule beside it. */}}
{{- define "automatos.validateEnterpriseRole" -}}
{{- $enterprise := .Values.edition.enterprise | default dict -}}
{{- $oidc := $enterprise.oidc | default dict -}}
{{- with toString ($enterprise.defaultRole | default "") | trim | lower }}
{{- if not (has . (list "none" "viewer" "editor" "admin")) }}
{{- fail "edition.enterprise.defaultRole must be none, viewer, editor or admin (ENTERPRISE_DEFAULT_ROLE): the API refuses to start otherwise" }}
{{- end }}
{{- if and (ne . "none") (not (toString ($oidc.requiredClaims | default "") | trim)) }}
{{- fail "edition.enterprise.defaultRole needs edition.enterprise.oidc.requiredClaims (ENTERPRISE_DEFAULT_ROLE, OIDC_REQUIRED_CLAIMS): with no access rule, everyone the identity provider lets sign in would get that role in the default workspace, so the API refuses to start. Set a rule, or set the role to none" }}
{{- end }}
{{- end }}
{{- end }}

{{/* The sign-in button's text, the scopes and the endpoint hosts, as the API checks them. */}}
{{- define "automatos.validateEnterpriseOidcText" -}}
{{- $oidc := (.Values.edition.enterprise | default dict).oidc | default dict -}}
{{- with toString ($oidc.displayName | default "") | trim }}
{{- if not (regexMatch `^[^\x00-\x1f\x7f-\x9f<>]{1,64}$` .) }}
{{- fail "edition.enterprise.oidc.displayName must be plain text of 1 to 64 characters, with no control character and no < or > (OIDC_DISPLAY_NAME): the API refuses to start otherwise" }}
{{- end }}
{{- end }}
{{- with toString ($oidc.scopes | default "") | trim }}
{{- $scopes := regexSplit `\s+` . -1 | compact }}
{{- $bad := not (has "openid" $scopes) }}
{{- range $scope := $scopes }}
{{- if not (regexMatch `^[\x21\x23-\x5B\x5D-\x7E]+$` $scope) }}{{ $bad = true }}{{ end }}
{{- end }}
{{- if $bad }}
{{- fail "edition.enterprise.oidc.scopes must be scope names separated by spaces, openid among them (OIDC_SCOPES): the API refuses to start otherwise" }}
{{- end }}
{{- end }}
{{- $label := `[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?` }}
{{- range $host := regexSplit `[\s,]+` (toString ($oidc.endpointHosts | default "")) -1 | compact }}
{{- $name := regexReplaceAll `\.+$` (lower $host) "" }}
{{- $ok := and (regexMatch (printf `^%s(?:\.%s)*$` $label $label) $name) (le (len $name) 253) }}
{{- if or (not $ok) (regexMatch `^[0-9]+$` (last (splitList "." $name))) }}
{{- fail "edition.enterprise.oidc.endpointHosts must be host names separated by commas, with no scheme, port, wildcard or address (OIDC_ENDPOINT_HOSTS): the API refuses to start otherwise" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
The enterprise API's settings, after the checks: nothing in any other edition. An optional
value left empty is not rendered, so the API's own default holds.
*/}}
{{- define "automatos.enterpriseApiEnv" -}}
{{- if include "automatos.enterprise" . }}
{{- include "automatos.validateEnterpriseConfig" . }}
{{- include "automatos.validateEnterpriseRequired" . }}
{{- include "automatos.validateEnterpriseValues" . }}
{{- $enterprise := .Values.edition.enterprise }}
{{- $oidc := $enterprise.oidc }}
- name: DEFAULT_WORKSPACE_ID
  value: {{ .Values.edition.defaultWorkspaceId | toString | trim | quote }}
- name: FRONTEND_URL
  value: {{ $enterprise.publicUrl | toString | trim | quote }}
- name: CLIENT_IP_HEADER
  value: {{ $enterprise.clientIpHeader | toString | trim | quote }}
- name: OIDC_ISSUER
  value: {{ $oidc.issuer | toString | trim | quote }}
- name: OIDC_CLIENT_ID
  value: {{ $oidc.clientId | toString | trim | quote }}
{{- range $setting := list (list "OIDC_DISPLAY_NAME" $oidc.displayName) (list "OIDC_SCOPES" $oidc.scopes) (list "OIDC_REQUIRED_CLAIMS" $oidc.requiredClaims) (list "OIDC_ENDPOINT_HOSTS" $oidc.endpointHosts) (list "ENTERPRISE_DEFAULT_ROLE" $enterprise.defaultRole) (list "ENTERPRISE_BOOTSTRAP_ADMINS" $enterprise.bootstrapAdmins) }}
{{- with index $setting 1 | default "" | toString | trim }}
- name: {{ index $setting 0 }}
  value: {{ . | quote }}
{{- end }}
{{- end }}
{{- if $oidc.caConfigMap }}
- name: OIDC_CA_BUNDLE_FILE
  value: {{ printf "%s/%s" (include "automatos.oidcCaDir" .) ($oidc.caKey | default "ca.crt") | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/* The identity provider's CA volume and its read-only mount: nothing without oidc.caConfigMap. */}}
{{- define "automatos.oidcCaVolume" -}}
{{- if include "automatos.enterprise" . }}
{{- with ((.Values.edition.enterprise | default dict).oidc | default dict).caConfigMap }}
- name: oidc-ca
  configMap:
    name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{- define "automatos.oidcCaMount" -}}
{{- if and (include "automatos.enterprise" .) ((.Values.edition.enterprise | default dict).oidc | default dict).caConfigMap }}
- name: oidc-ca
  mountPath: {{ include "automatos.oidcCaDir" . }}
  readOnly: true
{{- end }}
{{- end }}
