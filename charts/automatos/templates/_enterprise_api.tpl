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
DEFAULT_WORKSPACE_ID FRONTEND_URL CLIENT_IP_HEADER CLIENT_IP_PROXY_COUNT ENTERPRISE_BOOTSTRAP_ADMINS ENTERPRISE_DEFAULT_ROLE OIDC_ISSUER OIDC_CLIENT_ID OIDC_DISPLAY_NAME OIDC_SCOPES OIDC_REQUIRED_CLAIMS OIDC_ENDPOINT_HOSTS OIDC_CA_BUNDLE_FILE ENTERPRISE_CALLS_THEME_REVOCATIONS ENTERPRISE_CALLS_CLOUD_SYNC ENTERPRISE_CALLS_CLOUD_ITEMS ENTERPRISE_DEVICE_PAIRING
{{- end }}

{{/*
Each of Automatos's hosted services an enterprise install may call (PRD-277 US-704): its value under
edition.enterprise.hostedServices and the API setting it renders to, in that order. One per line.
*/}}
{{- define "automatos.hostedServices" -}}
themeRevocations ENTERPRISE_CALLS_THEME_REVOCATIONS
cloudSync ENTERPRISE_CALLS_CLOUD_SYNC
cloudItems ENTERPRISE_CALLS_CLOUD_ITEMS
{{- end }}

{{/*
edition.enterprise.hostedServices, as the API checks its settings at start (core/edition_start_up.py):
a map holding only the services above, each true or false (unset is false), and none switched on
without the one it needs: cloudSync needs themeRevocations (a theme brought from Automatos Cloud is
withdrawn only through that list), and cloudItems needs cloudSync (items arrive only in a Sync).
*/}}
{{- define "automatos.validateHostedServices" -}}
{{- $hosted := (.Values.edition.enterprise | default dict).hostedServices | default dict -}}
{{- if not (kindIs "map" $hosted) }}
{{- fail "edition.enterprise.hostedServices must be a map of themeRevocations, cloudSync and cloudItems, each true or false" }}
{{- end }}
{{- $known := list }}
{{- range $line := include "automatos.hostedServices" . | splitList "\n" }}
{{- $pair := splitList " " $line }}
{{- $known = append $known (index $pair 0) }}
{{- $value := index $hosted (index $pair 0) }}
{{- if not (or (kindIs "invalid" $value) (kindIs "bool" $value)) }}
{{- fail (printf "edition.enterprise.hostedServices.%s must be true or false (%s), written without quotes: the API refuses to start otherwise" (index $pair 0) (index $pair 1)) }}
{{- end }}
{{- end }}
{{- range $key := keys $hosted | sortAlpha }}
{{- if not (has $key $known) }}
{{- fail (printf "edition.enterprise.hostedServices.%s names no hosted service: the services are themeRevocations, cloudSync and cloudItems" $key) }}
{{- end }}
{{- end }}
{{- $on := dict }}
{{- range $name := $known }}
{{- $value := index $hosted $name }}
{{- $_ := set $on $name (and (kindIs "bool" $value) $value) }}
{{- end }}
{{- if and $on.cloudSync (not $on.themeRevocations) }}
{{- fail "edition.enterprise.hostedServices.cloudSync needs themeRevocations (ENTERPRISE_CALLS_CLOUD_SYNC, ENTERPRISE_CALLS_THEME_REVOCATIONS): a theme brought from Automatos Cloud is withdrawn only through Automatos's theme revocation list, so the API refuses to start. Switch both on, or leave cloudSync off" }}
{{- end }}
{{- if and $on.cloudItems (not $on.cloudSync) }}
{{- fail "edition.enterprise.hostedServices.cloudItems needs cloudSync (ENTERPRISE_CALLS_CLOUD_ITEMS, ENTERPRISE_CALLS_CLOUD_SYNC): agents, Playbooks, packages and plugins arrive from Automatos Cloud only in a Sync, so the API refuses to start. Switch both on, or leave cloudItems off" }}
{{- end }}
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

{{/*
edition.enterprise.clientIpProxyCount (CLIENT_IP_PROXY_COUNT) as trimmed text: empty when unset or
"". A 0 stays "0", so it is refused, never read as unset.
*/}}
{{- define "automatos.clientIpProxyCount" -}}
{{- $raw := (.Values.edition.enterprise | default dict).clientIpProxyCount -}}
{{- if not (kindIs "invalid" $raw) }}{{ toString $raw | trim }}{{ end -}}
{{- end }}

{{/* Stops the render on a `config` entry the enterprise edition sets elsewhere. */}}
{{- define "automatos.validateEnterpriseConfig" -}}
{{- $config := .Values.config | default dict -}}
{{- /* config.OIDC_CLIENT_SECRET is refused in every edition, by automatos.validateApi. */}}
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
{{- if hasKey $config "FORWARDED_ALLOW_IPS" }}
{{- fail "config.FORWARDED_ALLOW_IPS: in the enterprise edition the API reads the caller's address from edition.enterprise.clientIpHeader and clientIpProxyCount, and the server's own forwarded-header handling trusts only its own host. Remove it from config" }}
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
{{- if and (include "automatos.clientIpProxyCount" .) (not (toString ($enterprise.clientIpHeader | default "") | trim)) }}
{{- fail "edition.enterprise.clientIpProxyCount needs edition.enterprise.clientIpHeader (CLIENT_IP_PROXY_COUNT, CLIENT_IP_HEADER): name the header your own proxies append the caller's address to" }}
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
{{- range $address := list (list "edition.enterprise.oidc.issuer" $oidc.issuer) (list "edition.enterprise.publicUrl" $enterprise.publicUrl) }}
{{- if contains "@" (regexFind `^[^/?#]*//[^/?#]*` (index $address 1 | toString | trim)) }}
{{- fail (printf "%s must not hold a user name or password (an @ before the host): give the address alone" (index $address 0)) }}
{{- end }}
{{- end }}
{{- include "automatos.validateOidcCa" . }}
{{- if not (include "automatos.httpsAddress" $oidc.issuer) }}
{{- fail "edition.enterprise.oidc.issuer must be an https address with a host (OIDC_ISSUER): the API refuses to start otherwise" }}
{{- end }}
{{- if not (include "automatos.httpsAddress" $enterprise.publicUrl) }}
{{- fail "edition.enterprise.publicUrl must be an https address with a host (FRONTEND_URL): the API refuses to start otherwise" }}
{{- end }}
{{- include "automatos.validateClientIpProxyCount" . }}
{{- if not (regexMatch "^[!#$%&'*+.^_`|~0-9A-Za-z-]+$" (toString $enterprise.clientIpHeader | trim)) }}
{{- fail "edition.enterprise.clientIpHeader must be one header name, letters, digits and - (CLIENT_IP_HEADER): the API refuses to start otherwise" }}
{{- end }}
{{- include "automatos.validateEnterpriseRole" . }}
{{- include "automatos.validateEnterpriseOidcText" . }}
{{- include "automatos.validateHostedServices" . }}
{{- if not (or (kindIs "invalid" $enterprise.devicePairing) (kindIs "bool" $enterprise.devicePairing)) }}
{{- fail "edition.enterprise.devicePairing must be true or false (ENTERPRISE_DEVICE_PAIRING), written without quotes: the API refuses to start otherwise" }}
{{- end }}
{{- end }}

{{/*
The provider's CA: a ConfigMap name Kubernetes accepts (a DNS subdomain: lower-case letters,
digits, - and ., at most 253), and a key that is a plain file name, with no path separator and
no "..", since it names the file under the mount.
*/}}
{{- define "automatos.validateOidcCa" -}}
{{- $oidc := (.Values.edition.enterprise | default dict).oidc | default dict -}}
{{- with $oidc.caConfigMap }}
{{- $name := toString . }}
{{- $label := `[a-z0-9]([-a-z0-9]*[a-z0-9])?` }}
{{- if or (gt (len $name) 253) (not (regexMatch (printf `^%s(\.%s)*$` $label $label) $name)) }}
{{- fail "edition.enterprise.oidc.caConfigMap must be a ConfigMap's name: lower-case letters, digits, - and ., at most 253 characters, starting and ending with a letter or digit" }}
{{- end }}
{{- $key := toString ($oidc.caKey | default "ca.crt") }}
{{- if or (gt (len $key) 253) (not (regexMatch `^[-._a-zA-Z0-9]+$` $key)) (contains ".." $key) }}
{{- fail "edition.enterprise.oidc.caKey must be a plain file name, the ConfigMap's key: letters, digits, -, _ and ., with no path separator and no \"..\"" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
How many of the install's own proxies append to the client address header, as the API checks it
(core/client_ip_settings.py): a plain whole number from 1 to 8 when set (no sign, point or leading
zero in its text; YAML has already read a number written unquoted), and set when the header is
one proxies append to. The header may not be RFC 7239's Forwarded, whose values are parameters.
*/}}
{{- define "automatos.validateClientIpProxyCount" -}}
{{- $header := toString ((.Values.edition.enterprise | default dict).clientIpHeader | default "") | trim | lower -}}
{{- $count := include "automatos.clientIpProxyCount" . -}}
{{- if and $count (or (not (regexMatch `^[1-9][0-9]{0,2}$` $count)) (gt (atoi $count) 8)) }}
{{- fail "edition.enterprise.clientIpProxyCount must be a plain whole number from 1 to 8 (CLIENT_IP_PROXY_COUNT): how many of your own proxies append the caller's address to edition.enterprise.clientIpHeader. The API refuses to start otherwise" }}
{{- end }}
{{- if eq $header "forwarded" }}
{{- fail "edition.enterprise.clientIpHeader must name a header that holds one IP address, or IP addresses separated by commas, as X-Real-IP and X-Forwarded-For do (CLIENT_IP_HEADER): RFC 7239's for= parameters are not read, so the API refuses to start otherwise" }}
{{- end }}
{{- if and (not $count) (has $header (list "x-forwarded-for" "x-original-forwarded-for")) }}
{{- fail "edition.enterprise.clientIpProxyCount must be set (CLIENT_IP_PROXY_COUNT): edition.enterprise.clientIpHeader names a header proxies append to, so say how many of your own proxies append to it (1 for one load balancer, 2 for a content network in front of one)" }}
{{- end }}
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
{{- with include "automatos.clientIpProxyCount" . }}
- name: CLIENT_IP_PROXY_COUNT
  value: {{ . | quote }}
{{- end }}
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
{{- $hosted := $enterprise.hostedServices | default dict }}
{{- range $line := include "automatos.hostedServices" . | splitList "\n" }}
{{- $pair := splitList " " $line }}
- name: {{ index $pair 1 }}
  value: {{ eq (toString (index $hosted (index $pair 0))) "true" | ternary "true" "false" | quote }}
{{- end }}
- name: ENTERPRISE_DEVICE_PAIRING
  value: {{ kindIs "invalid" $enterprise.devicePairing | ternary "true" (toString $enterprise.devicePairing) | quote }}
{{- end }}
{{- end }}

{{/* The identity provider's CA volume and its read-only mount: nothing without oidc.caConfigMap. */}}
{{- define "automatos.oidcCaVolume" -}}
{{- if include "automatos.enterprise" . }}
{{- with ((.Values.edition.enterprise | default dict).oidc | default dict).caConfigMap }}
- name: oidc-ca
  configMap:
    name: {{ . | toString | quote }}
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
