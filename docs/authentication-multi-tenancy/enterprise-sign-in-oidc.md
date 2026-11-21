# Enterprise Sign-In with OIDC

<details>
<summary>Relevant source files</summary>

The following files were used as context for this page:

- [orchestrator/config.py](../../orchestrator/config.py)
- [orchestrator/core/edition_start_up.py](../../orchestrator/core/edition_start_up.py)
- [orchestrator/core/auth/oidc_provider.py](../../orchestrator/core/auth/oidc_provider.py)
- [orchestrator/core/auth/oidc_sign_in.py](../../orchestrator/core/auth/oidc_sign_in.py)
- [orchestrator/core/auth/oidc_tokens.py](../../orchestrator/core/auth/oidc_tokens.py)
- [orchestrator/core/auth/oidc_exchange.py](../../orchestrator/core/auth/oidc_exchange.py)
- [orchestrator/core/auth/access_rule.py](../../orchestrator/core/auth/access_rule.py)
- [orchestrator/core/auth/bootstrap_admins.py](../../orchestrator/core/auth/bootstrap_admins.py)
- [orchestrator/core/auth/first_admin.py](../../orchestrator/core/auth/first_admin.py)
- [orchestrator/core/auth/sign_in_flow.py](../../orchestrator/core/auth/sign_in_flow.py)
- [orchestrator/core/auth/step_up.py](../../orchestrator/core/auth/step_up.py)
- [orchestrator/core/auth/step_up_sign_in.py](../../orchestrator/core/auth/step_up_sign_in.py)
- [orchestrator/core/auth/sessions.py](../../orchestrator/core/auth/sessions.py)
- [orchestrator/core/auth/session_cookie.py](../../orchestrator/core/auth/session_cookie.py)
- [orchestrator/core/auth/session_audit.py](../../orchestrator/core/auth/session_audit.py)
- [orchestrator/core/auth/phone_sign_in.py](../../orchestrator/core/auth/phone_sign_in.py)
- [orchestrator/core/auth/sign_in_start_up.py](../../orchestrator/core/auth/sign_in_start_up.py)
- [orchestrator/api/auth_oidc.py](../../orchestrator/api/auth_oidc.py)
- [orchestrator/api/auth_sessions.py](../../orchestrator/api/auth_sessions.py)
- [charts/automatos/values.yaml](../../charts/automatos/values.yaml)
- [charts/automatos/templates/_enterprise_api.tpl](../../charts/automatos/templates/_enterprise_api.tpl)
- [docs/PRDS/PRD-277-ENTERPRISE-OIDC-SIGN-IN.md](../PRDS/PRD-277-ENTERPRISE-OIDC-SIGN-IN.md)

</details>

This page explains how the enterprise edition (`AUTH_EDITION=enterprise`) signs people in with an organisation's OpenID Connect (OIDC) provider. It covers what the provider must offer, which claims Automatos reads, and every setting involved. It is for the person who registers Automatos at the provider and fills in the Helm chart values.

Each provider has its own setup guide:

- [Microsoft Entra ID](enterprise-sign-in-entra-id.md)
- [Google Workspace](enterprise-sign-in-google-workspace.md)
- [Okta](enterprise-sign-in-okta.md)
- [Keycloak](enterprise-sign-in-keycloak.md)
- [Amazon Cognito](enterprise-sign-in-amazon-cognito.md)
- [AWS IAM Identity Center](enterprise-sign-in-aws-iam-identity-center.md): not supported as an OIDC provider. Its page gives AWS's reason and the path AWS documents instead.

The requirements are in [PRD-277](../PRDS/PRD-277-ENTERPRISE-OIDC-SIGN-IN.md). The chart's own instructions are in the [chart README](../../charts/automatos/README.md) and in [Kubernetes & Helm Chart](../deployment-infrastructure/kubernetes-helm-chart.md).

---

## The flow

Automatos's backend is the OIDC client. It is a confidential client: it holds a client secret.

1. The browser opens `GET /api/v1/auth/oidc/start`.
2. Automatos reads the provider's metadata from `<OIDC_ISSUER>/.well-known/openid-configuration`.
3. Automatos sends the browser to the provider's `authorization_endpoint` with:
   - `response_type=code`
   - `client_id`, `redirect_uri` and `scope`
   - `state` and `nonce`
   - `code_challenge` with `code_challenge_method=S256` (PKCE)
   - `claims={"id_token":{"auth_time":{"essential":true}}}` (see [auth_time](#auth_time-and-step-up))
4. The person signs in at the provider.
5. The provider sends the browser to `<install origin>/api/v1/auth/oidc/callback` with a code.
6. Automatos exchanges the code at the provider's `token_endpoint`, with the PKCE verifier and the client secret.
7. Automatos verifies the ID token, applies the access rule, finds or makes the person, and issues a session.

Automatos uses the ID token only. It does not call the userinfo endpoint and keeps no provider access or refresh token.

### What the provider's metadata must offer

The API reads the metadata at start and refuses metadata it cannot use [orchestrator/core/auth/oidc_provider.py:235-255]():

- Its `issuer` equals `OIDC_ISSUER` exactly.
- Every endpoint is an `https` address.
- The `token_endpoint` and the `jwks_uri` are on the issuer's host, or on a host listed in `OIDC_ENDPOINT_HOSTS`. The `authorization_endpoint` is opened by the browser, so its host is not limited [orchestrator/core/auth/oidc_provider.py:173-176]().
- It offers the response type `code`, the PKCE method `S256`, and one of the signing algorithms `RS256`, `PS256` or `ES256` [orchestrator/core/auth/oidc_provider.py:69](). A list the metadata leaves out counts as offering everything.
- It offers client authentication with a secret. Automatos uses `client_secret_basic` when it is listed or the list is absent, else `client_secret_post` [orchestrator/core/auth/oidc_provider.py:197-204](). The provider must issue the application a client secret.

### The redirect address

Register exactly one sign-in redirect address at the provider:

```
<FRONTEND_URL origin>/api/v1/auth/oidc/callback
```

The origin is the scheme, host and port of `FRONTEND_URL` (the chart's `edition.enterprise.publicUrl`), in lower case, with a default port left out [orchestrator/core/auth/oidc_sign_in.py:83](), [orchestrator/core/auth/oidc_sign_in.py:101-106](). For `publicUrl: https://automatos.example.com` it is:

```
https://automatos.example.com/api/v1/auth/oidc/callback
```

Register the address after sign-out as well, where the provider asks for one:

```
https://automatos.example.com/sign-in
```

The install notes the chart prints show both addresses.

### Scopes

`OIDC_SCOPES` defaults to `openid profile email`. It must include `openid`; the API refuses to start otherwise. Add a scope only when a provider needs one to put a claim in the ID token, for example Okta's `groups`.

---

## The ID token

### What is checked

Every ID token must pass these checks [orchestrator/core/auth/oidc_tokens.py:1-43]():

| Check | Rule |
|---|---|
| Size | At most 16 KiB. |
| Algorithm | `RS256`, `PS256` or `ES256`, and the header names a `kid`. |
| Signature | Verified against the provider's `jwks_uri`. An unknown `kid` makes Automatos read the keys again once. |
| `iss` | Equals `OIDC_ISSUER`. |
| `aud` | Contains `OIDC_CLIENT_ID`. |
| `azp` | Required when `aud` holds more than one value. When present, it equals `OIDC_CLIENT_ID`. |
| `exp`, `iat` | Required. `exp`, `iat` and `nbf` are checked with 60 seconds of leeway. |
| `nonce` | Equals the one sent at the start. |
| `sub` | A non-empty string. |

A token that fails any check is refused with `sign_in_refused`, audited with the reason `invalid_token`.

### The claims Automatos reads

| Claim | Setting that names it | Required | Used for | When it is missing |
|---|---|---|---|---|
| `iss` + subject | `OIDC_CLAIM_SUBJECT` (default `sub`) | Yes | The person's identity: the pair (issuer, subject). | `sign_in_refused`. |
| `email` | `OIDC_CLAIM_EMAIL` (default `email`) | No | The person's address, kept only when verified. | No address is kept. |
| `email_verified` | none | No | Whether `email` is verified. Only the JSON `true` counts. | The address counts as unverified and is not kept. |
| `name` | `OIDC_CLAIM_NAME` (default `name`) | No | The name shown in Automatos. | No name is shown from the provider. |
| `auth_time` | none | For step-up and phones | When the person last signed in at the provider. | The session records 1 January 1970. Every step-up is refused with `step_up_no_auth_time`, and every phone sign-in with `sign_in_refused`. |
| `sid` | none | No | Kept with the session when it fits. | Nothing changes. |
| Any claim the access rule names | `OIDC_REQUIRED_CLAIMS` | When a rule names it | Who may sign in. | `sign_in_not_allowed`. |

Sources: [orchestrator/core/auth/oidc_sign_in.py:173-234]().

### Identity across sign-ins

A person is found by the pair (issuer, subject), never by email address [orchestrator/core/auth/oidc_sign_in.py:186-196]().

- `OIDC_CLAIM_SUBJECT` names the subject claim. It defaults to `sub`.
- Choose it before the first sign-in. Changing it later makes everyone's next sign-in a first sign-in. A person already linked under `sub` is then refused, audited `subject_claim_changed`.
- The claim must be unique to one person, never change, never be reused, and not be editable by the person.
- The API refuses to start when `OIDC_CLAIM_SUBJECT` names a claim that cannot be that: a protocol claim, `email_verified`, `acr`, `amr`, `ver`, `tid`, `hd`, `groups`, `roles`, `picture`, `locale`, `preferred_username`, `upn`, `email`, or the claim `OIDC_CLAIM_EMAIL` or `OIDC_CLAIM_NAME` names [orchestrator/core/auth/access_rule.py:82-91]().

### The email address

- The address is kept only when it is verified: the claim read is `email` and `email_verified` is `true` [orchestrator/core/auth/oidc_sign_in.py:173-183]().
- An address read from another claim counts as unverified, unless `OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES=true`. Set that only when the provider puts nothing but addresses it has verified in that claim.
- A verified address already held by another person refuses the sign-in: `sign_in_refused`, audited `email_conflict`.

---

## Who may sign in

### The access rule

`OIDC_REQUIRED_CLAIMS` holds rules a token must match before anyone is let in [orchestrator/core/auth/access_rule.py:8-54]().

- Each rule is `name=value`. Rules are separated by `;`. A rule is split at its first `=`.
- A claim matches when it is the exact string, or a list that contains the string.
- Names are read literally, so `cognito:groups` and `automatos:team` work.
- At most 32 rules. A name is 1 to 256 characters; a value is 1 to 1024.
- A rule may not name `iss`, `aud`, `azp`, `exp`, `iat`, `nbf`, `nonce`, `auth_time`, `at_hash`, `c_hash`, `sid` or `jti`.

A token that fails the rule is refused with `sign_in_not_allowed`, audited `access_rule`. No person is made.

The rule is required in two cases. The API refuses to start without one:

- The issuer is shared by many organisations: `https://accounts.google.com`, or Microsoft's `common`, `organizations` or `consumers` authorities or the personal-accounts tenant `9188040d-6c67-4c5b-b112-36a304b66dad` [orchestrator/core/auth/access_rule.py:74-79]().
- `ENTERPRISE_DEFAULT_ROLE` is anything but `none`.

Example: `OIDC_REQUIRED_CLAIMS=hd=example.com` lets in only accounts in the `example.com` Google Workspace domain.

### The first admin

`ENTERPRISE_BOOTSTRAP_ADMINS` names the people who become install admins [orchestrator/core/auth/bootstrap_admins.py](), [orchestrator/core/auth/first_admin.py]().

- Entries are `sub:<subject>` or `email:<address>`, separated by commas.
- At most 50 entries, each at most 320 characters. No `*` or `?`.
- `sub:` matches the subject Automatos reads, the claim `OIDC_CLAIM_SUBJECT` names.
- `email:` matches only the standard `email` claim with `email_verified` `true`. The domain is compared without regard to case.
- The grant happens at a person's first sign-in, or at any sign-in while the install has no active install admin.
- The person gets `system_role=admin`. They become owner of the default workspace when it has no owner, and admin of it otherwise. The grant is audited `auth:bootstrap_admin`.

Use `sub:` for a provider that sends no `email_verified`.

### The default role

`ENTERPRISE_DEFAULT_ROLE` is the role in the default workspace (`DEFAULT_WORKSPACE_ID`) for anyone who passes the access rule. It is `none`, `viewer`, `editor` or `admin`, and `none` by default. It applies at a person's first sign-in only.

A person in no workspace is answered `403 workspace_required` until an admin adds them to one.

### Access by group or domain

Groups matter only for the access rule. Roles are set in Automatos. What each provider sends:

| Provider | Claim for a rule | Notes |
|---|---|---|
| Microsoft Entra ID | `roles` (app roles), `groups`, `tid` | Above 200 groups, Entra leaves `groups` out of the token. A rule on an app role is not affected. |
| Google Workspace | `hd` | Google sends no groups. Access by group is not available with Google. |
| Okta | `groups`, once a groups claim is configured | A groups filter matching more than 100 groups fails the request at Okta. |
| Keycloak | Any claim a protocol mapper adds | The CI realm adds `automatos:team`. |
| Amazon Cognito | `cognito:groups` | A user belongs to at most 100 groups. |

---

## Sessions

A sign-in issues an Automatos session [orchestrator/core/auth/sessions.py:1-35](), [orchestrator/core/auth/session_cookie.py:1-30]().

- The session token is `ams_` followed by 32 random bytes. It is held in the cookie `__Host-automatos_session`: `Secure`, `HttpOnly`, `SameSite=Lax`, `Path=/`.
- Two clocks end a session:
  - `SESSION_IDLE_MINUTES`, 120 by default, at least 5, and no longer than the absolute clock. Use moves it forward; the session's row is written at most every 5 minutes.
  - `SESSION_ABSOLUTE_HOURS`, 12 by default. It never moves.
- Automatos does not renew a session from the provider. It keeps no refresh token. When a session ends, the person signs in again, and the provider decides whether to ask for a password.
- A new session is issued at every sign-in.
- A change made with the session cookie needs the same `Origin` and the header `X-Automatos-CSRF`. Otherwise it is answered `403 cross_site_request_refused`.
- An ended session is answered `401 session_ended`, with the reason `unknown`, `revoked` or `expired`.

---

## `auth_time` and step-up

### Step-up

Admin actions need a recent sign-in [orchestrator/core/auth/step_up.py:85-90](), [orchestrator/core/auth/step_up_sign_in.py:1-45]().

1. The session's `auth_time` is older than `STEP_UP_MAX_AGE_MINUTES` (10 by default). The API answers `403 step_up_required` and names `/api/v1/auth/oidc/start?step_up=1`.
2. The browser opens that address. It is accepted only when the browser's `Sec-Fetch-Site` is `same-origin` or `none`.
3. Automatos asks the provider for a fresh sign-in: `max_age=0`, plus `prompt=login` where the provider takes it (below).
4. The callback reads `auth_time` from the new ID token. It confirms the step-up only when that time is no earlier than the step-up's start, less 60 seconds, and not in the future.
5. A confirmed step-up issues a new session. The new session keeps the old one's absolute expiry.

A step-up is refused with:

- `step_up_no_auth_time` when the ID token carries no `auth_time`;
- `step_up_refused` when `auth_time` is older than the start, in the future, or when another person signed in, the session ended, or the browser did not present the session.

The audit reasons are `no_auth_time`, `stale_auth_time`, `future_auth_time`, `another_person`, `session_ended`, `session_not_presented` and `deactivated` [orchestrator/core/auth/session_audit.py:84-96]().

### Why `auth_time` is needed

`max_age=0` and `prompt=login` only ask the provider to make the person sign in again. Automatos cannot see whether it did. The ID token's `auth_time` is the only proof of when the person last signed in. So the step-up decision is read from `auth_time` alone.

An ordinary sign-in whose ID token has no `auth_time` still works. Its session records 1 January 1970 [orchestrator/core/auth/oidc_sign_in.py:91](). Every admin action then needs a step-up, and every step-up is refused with `step_up_no_auth_time`. So set up the provider to send `auth_time` before the install is used.

### What Automatos asks for

[orchestrator/core/auth/oidc_sign_in.py:114-126](), [orchestrator/core/auth/oidc_provider.py:207-232]()

| Parameter | Sent when |
|---|---|
| `claims={"id_token":{"auth_time":{"essential":true}}}` | Every start, unless the metadata has `claims_parameter_supported` with any value but `true`. |
| `max_age=0` | A step-up, or a phone's start. |
| `prompt=login` | A step-up, or a phone's start, when the metadata's `prompt_values_supported` lists `login`. Also when it lists no values, except for the issuer host `accounts.google.com`. |

---

## Sign-out

`POST /api/v1/auth/sign-out` ends the Automatos session [orchestrator/api/auth_sessions.py](), [orchestrator/core/auth/oidc_sign_in.py:266-293]().

- It answers `204` when there is nowhere else to go.
- It answers `200` with `{"redirect_to": "<address>"}` when the provider's metadata lists an `end_session_endpoint` on the issuer's host or a host in `OIDC_ENDPOINT_HOSTS`. The browser then goes there.
- That address carries exactly two parameters: `client_id` and `post_logout_redirect_uri=<install origin>/sign-in`.
- No `id_token_hint` is sent, because Automatos keeps no ID token.

What happens at the provider depends on the provider. Each guide's **Sign-out** section says what its documentation describes. The Automatos session has ended in every case.

---

## Phones

Automatos Go signs in through the same start address. Nothing extra is registered at the provider [orchestrator/core/auth/phone_sign_in.py:40-75](), [orchestrator/api/auth_oidc.py:1-55]().

1. The app opens `/api/v1/auth/oidc/start` with `app_link=go`, `code_challenge`, `code_challenge_method=S256`, `install_id` and `platform` (`ios` or `android`). A malformed start is answered `400 invalid_phone_start`.
2. Automatos asks the provider for a fresh sign-in (`max_age=0`, and `prompt=login` where taken).
3. The callback requires `auth_time` no earlier than the phone's start. Otherwise the app link is answered `error=sign_in_refused`, audited `no_auth_time`, `stale_auth_time` or `future_auth_time`.
4. The browser shows the confirm page `/sign-in/phone`. The person presses Continue within 5 minutes.
5. The app link `automatos://go/signed-in?code=…` opens the app.
6. The app exchanges the code at `POST /api/v1/mobile/sign-in/exchange`. A bad code is answered `401 invalid_sign_in_code`.

The confirm page shows `no_workspace`, `several_workspaces` or `device_limit_reached` when the person cannot pair a phone.

`ENTERPRISE_DEVICE_PAIRING` (`true` by default) switches phone pairing on or off. `ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS` (90 by default, 1 to 3650) is how long a phone's token lives after its last pairing or sign-in. An expired token is answered `401 device_token_expired`. The app finds the install through `/.well-known/automatos.json`.

A provider that never sends a fresh `auth_time` refuses every phone sign-in. Such a phone pairs with a code from Settings instead.

---

## Settings

Every sign-in setting, with where it is set in the chart. The chart sets the `edition.enterprise` ones; `config:` may not set them. See [charts/automatos/values.yaml](../../charts/automatos/values.yaml) and [orchestrator/config.py:601-638]().

| Environment name | Chart value | Default | Notes |
|---|---|---|---|
| `AUTH_EDITION` | `edition.authEdition` | `local` | `enterprise`. |
| `FRONTEND_URL` | `edition.enterprise.publicUrl` | none, required | `https://<host>`. Gives the redirect origin. |
| `DEFAULT_WORKSPACE_ID` | `edition.defaultWorkspaceId` | `00000000-0000-0000-0000-0000000000c1` | A UUID. |
| `CLIENT_IP_HEADER` | `edition.enterprise.clientIpHeader` | none, required | For example `X-Real-IP`. |
| `CLIENT_IP_PROXY_COUNT` | `edition.enterprise.clientIpProxyCount` | empty | 1 to 8. Required with `X-Forwarded-For`. |
| `OIDC_ISSUER` | `edition.enterprise.oidc.issuer` | none, required | `https` only. |
| `OIDC_CLIENT_ID` | `edition.enterprise.oidc.clientId` | none, required | |
| `OIDC_CLIENT_SECRET` | the Secret `existingSecret` names, key `OIDC_CLIENT_SECRET` | none, required | Never a chart value. |
| `OIDC_DISPLAY_NAME` | `edition.enterprise.oidc.displayName` | `your organisation` | The sign-in button's text, 1 to 64 characters. |
| `OIDC_SCOPES` | `edition.enterprise.oidc.scopes` | `openid profile email` | Must include `openid`. |
| `OIDC_REQUIRED_CLAIMS` | `edition.enterprise.oidc.requiredClaims` | empty | The access rule. |
| `OIDC_ENDPOINT_HOSTS` | `edition.enterprise.oidc.endpointHosts` | empty | Host names, separated by commas. |
| `OIDC_CA_BUNDLE_FILE` | `edition.enterprise.oidc.caConfigMap` and `caKey` | empty | For a provider with a private certificate authority. |
| `ENTERPRISE_BOOTSTRAP_ADMINS` | `edition.enterprise.bootstrapAdmins` | empty | `sub:` and `email:` entries. |
| `ENTERPRISE_DEFAULT_ROLE` | `edition.enterprise.defaultRole` | `none` | Needs an access rule when not `none`. |
| `ENTERPRISE_DEVICE_PAIRING` | `edition.enterprise.devicePairing` | `true` | |
| `ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS` | `edition.enterprise.deviceTokenMaxAgeDays` | `90` | 1 to 3650. |
| `OIDC_CLAIM_SUBJECT` | `config:` | `sub` | Choose before the first sign-in. |
| `OIDC_CLAIM_EMAIL` | `config:` | `email` | |
| `OIDC_CLAIM_NAME` | `config:` | `name` | |
| `OIDC_CLAIM_EMAIL_HOLDS_ONLY_VERIFIED_ADDRESSES` | `config:` | `false` | |
| `SESSION_IDLE_MINUTES` | `config:` | `120` | At least 5. |
| `SESSION_ABSOLUTE_HOURS` | `config:` | `12` | |
| `STEP_UP_MAX_AGE_MINUTES` | `config:` | `10` | |
| `OIDC_PENDING_SIGN_INS_MAX` | `config:` | `10000` | Unfinished sign-ins held at once. |
| `OIDC_PENDING_SIGN_INS_PER_ADDRESS` | `config:` | `100` | Unfinished sign-ins from one client address. |

The session cookie's name (`SESSION_COOKIE_NAME`) and the CSRF header's (`SESSION_CSRF_HEADER`) are fixed in the enterprise web image. `config:` may not change them.

---

## Error codes

### On the sign-in page

The callback sends the browser to `/sign-in?error=<code>` [orchestrator/core/auth/sign_in_flow.py:123-130]():

| Code | Meaning | Audit reason |
|---|---|---|
| `sign_in_expired` | The sign-in took too long, or was started in another tab. | |
| `sign_in_refused` | The ID token failed a check, its email belongs to another person, the person is deactivated, or the subject claim changed. | `invalid_token`, `email_conflict`, `deactivated`, `subject_claim_changed` |
| `sign_in_not_allowed` | The token did not match the access rule. | `access_rule` |
| `step_up_refused` | A step-up did not prove a fresh sign-in by the same person. | see [Step-up](#step-up) |
| `step_up_no_auth_time` | The provider's ID token has no `auth_time`. | `no_auth_time` |
| `identity_provider_unavailable` | The provider could not be reached, or its metadata or keys cannot be used. | |
| `sign_in_failed` | Any other failure. | |

### From the API

| Status and code | Meaning |
|---|---|
| `401 session_ended` | The session is unknown, revoked or expired. |
| `403 workspace_required` | The person is in no workspace. |
| `403 step_up_required` | The action needs a recent sign-in. |
| `403 cross_site_request_refused` | A cookie-authenticated change came without the same origin or the CSRF header. |
| `429` | More than 300 starts, or 300 callbacks, from one address in 10 minutes. |
| `400 invalid_phone_start` | A phone's start had a missing or malformed parameter. |
| `401 invalid_sign_in_code` | A phone's code was refused. Any code found is spent. |
| `401 device_token_expired` | A phone's token is older than `ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS`. |

### At start-up

The API refuses to start, with a message naming the setting, when:

- `OIDC_ISSUER`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET` or `FRONTEND_URL` is unset: `AUTH_EDITION=enterprise cannot start: <names> <is|are> unset.`
- any other setting is malformed: `AUTH_EDITION=enterprise cannot start: <name> <problem>.`

The chart stops rendering on most of the same mistakes, before anything is deployed.

A provider that cannot be reached does not stop the API. `/health/bootstrap` shows the `oidc_provider` stage as failed, and sign-in answers `identity_provider_unavailable` [orchestrator/core/auth/sign_in_start_up.py:1-36](). The chart's ingress does not route `/health`; read it from inside the cluster.

---

## Check a registration works

1. Install the chart. The API starts, and the install notes print the callback and sign-out addresses.
2. Open `/health/bootstrap` from inside the cluster. The `oidc_provider` stage is not marked failed.
3. Open the install's address. The button reads **Sign in with** and the `displayName`.
4. Sign in as a person the access rule admits. You reach Automatos.
5. Sign in as a person the access rule refuses. You see `sign_in_not_allowed`.
6. Sign in as a first admin. Wait longer than `STEP_UP_MAX_AGE_MINUTES`. Change a person's role on the Team page. Automatos sends you to the provider to sign in again, and the change goes through. A refusal with `step_up_no_auth_time` means the provider sends no `auth_time`.
7. Sign out. You land on `/sign-in`. Note what the provider shows.
8. Optional: sign in on Automatos Go. The confirm page appears, and the app opens signed in.

---

## Provider comparison

From PRD-277 §5.9 and each provider's documentation, read on 21 November 2026. None of the rows except Keycloak's has been run against a live tenant; the conformance run (US-703) records that.

| Provider | Issuer | Groups in the ID token | `auth_time` | Endpoint hosts to list | Sign-out at the provider |
|---|---|---|---|---|---|
| Microsoft Entra ID | `https://login.microsoftonline.com/<tenant-id>/v2.0` | Optional; left out above 200. App roles in `roles`. | Add the `auth_time` optional claim. | none | `end_session_endpoint` on the issuer's host. `post_logout_redirect_uri` must be a registered redirect URI. |
| Google Workspace | `https://accounts.google.com` | None | Enable **Session age claims** for the project; the app must be published and verified. | `oauth2.googleapis.com,www.googleapis.com` | No `end_session_endpoint`: Automatos only. |
| Okta | `https://<okta-domain>` or `https://<okta-domain>/oauth2/<server-id>` | Through a groups claim; more than 100 matches fails the request. | Always present. | none | Okta's logout reference lists `id_token_hint` as required; Automatos sends `client_id` and `post_logout_redirect_uri`. |
| Keycloak | `https://<host>/realms/<realm>` | Through a protocol mapper. | In the metadata's claims; CI checks a step-up. | none | Accepts `client_id` and `post_logout_redirect_uri` (CI checks it). |
| Amazon Cognito | `https://cognito-idp.<region>.amazonaws.com/<pool-id>`, or the `issuer-cognito-idp` form | `cognito:groups` | Listed among the claims; a step-up needs managed login. | The user pool domain | Cognito's `/logout` takes `logout_uri`, not `post_logout_redirect_uri`. |
| AWS IAM Identity Center | not applicable | not applicable | not applicable | not applicable | Not supported as an OIDC provider: see [its page](enterprise-sign-in-aws-iam-identity-center.md). |
