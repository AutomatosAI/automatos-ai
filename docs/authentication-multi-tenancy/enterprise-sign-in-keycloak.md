# Enterprise Sign-In: Keycloak

How to register Automatos with Keycloak for the enterprise edition. Read [Enterprise Sign-In with OIDC](enterprise-sign-in-oidc.md) first: it explains the flow, the claims and every setting.

## 1. Evidence

- **Exercised by CI.** The workflow [`.github/workflows/oidc-keycloak.yml`](../../.github/workflows/oidc-keycloak.yml), job `keycloak` ("Sign-in against a real Keycloak (non-required)"), starts Keycloak `26.8.2` and runs [`orchestrator/tests/test_prd277_w4_keycloak_sign_in.py`](../../orchestrator/tests/test_prd277_w4_keycloak_sign_in.py) against it.
- **The realm file.** No realm file is committed. The workflow's step "Make the certificate, the credentials and the realm" writes `realm.json` at run time, with a generated client secret and password, and Keycloak imports it at start (`--import-realm`). [Register the application](#2-register-the-application) shows that realm's client.
- **What the job checks:**
  - a person signs in through Keycloak's own login form and gets a session;
  - the access rule reads a claim with a colon in its name (`automatos:team`): `=finance` is refused, `=ops` is admitted;
  - a step-up sends `max_age=0`, `prompt=login` and the `claims` parameter, Keycloak asks for the password again, and a new session is issued;
  - sign-out sends the browser to Keycloak's end-session address with `client_id` and `post_logout_redirect_uri`, and Keycloak accepts it.
- The job is not a required check. PRD-277 US-407 records whether it has been seen green.
- Keycloak's documentation, read on 21 November 2026:
  - [Securing applications: OIDC layers](https://www.keycloak.org/securing-apps/oidc-layers)
  - [Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/index.html) (version 26.8.2)

## 2. Register the application

Create a confidential OpenID Connect client in the realm people sign in to. The CI realm's client is the worked example. Its settings, as Keycloak's realm JSON names them:

```json
{
  "clientId": "automatos",
  "enabled": true,
  "protocol": "openid-connect",
  "publicClient": false,
  "clientAuthenticatorType": "client-secret",
  "standardFlowEnabled": true,
  "directAccessGrantsEnabled": false,
  "implicitFlowEnabled": false,
  "redirectUris": ["https://automatos.example.com/api/v1/auth/oidc/callback"],
  "attributes": {
    "pkce.code.challenge.method": "S256",
    "post.logout.redirect.uris": "https://automatos.example.com/sign-in"
  }
}
```

What each setting does:

| Realm JSON | Meaning |
|---|---|
| `publicClient: false`, `clientAuthenticatorType: client-secret` | A confidential client that authenticates with a client secret. |
| `standardFlowEnabled: true` | The authorization code flow. |
| `directAccessGrantsEnabled: false`, `implicitFlowEnabled: false` | No password grant and no implicit flow. Automatos uses neither. |
| `redirectUris` | The sign-in callback, exactly. |
| `pkce.code.challenge.method: S256` | PKCE with `S256`, as Automatos sends it. |
| `post.logout.redirect.uris` | The address after sign-out. Keycloak's guide calls it **Valid Post Logout Redirect URIs**. |

You can make the client in the Admin Console or import it with the realm. The console's labels for the other fields are not confirmed here; see the [Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/index.html).

Then:

1. Copy the client's secret from Keycloak.
2. Put it in the Kubernetes Secret that `existingSecret` names, under the key `OIDC_CLIENT_SECRET`.

## 3. The issuer address

```
https://<keycloak-host>/realms/<realm>
```

- The metadata address is `https://<keycloak-host>/realms/<realm>/.well-known/openid-configuration`.
- The token endpoint (`…/protocol/openid-connect/token`) and the keys (`…/protocol/openid-connect/certs`) are under the realm's address. Leave `oidc.endpointHosts` empty.
- A Keycloak with a certificate from a private certificate authority: put the authority's certificate in a ConfigMap, and set `oidc.caConfigMap` and `oidc.caKey`. The CI job signs in this way, with a certificate it makes for `localhost` and `OIDC_CA_BUNDLE_FILE` naming it.

## 4. Claims

| Claim | Source | Automatos setting |
|---|---|---|
| `sub` | Keycloak's id for the user. | `OIDC_CLAIM_SUBJECT` default (`sub`). |
| `email`, `email_verified` | The user's email and its verified flag. The CI user has `emailVerified: true`. | Defaults. |
| `name` | From the user's profile. | `OIDC_CLAIM_NAME` default. |
| `auth_time` | See [`auth_time`](#5-auth_time). | Nothing to set. |
| Any claim a protocol mapper adds | A mapper on the client. | Access rule: `<claim>=<value>`. |

The CI realm adds a claim with a protocol mapper of type `oidc-hardcoded-claim-mapper`:

```json
{
  "name": "automatos-team",
  "protocol": "openid-connect",
  "protocolMapper": "oidc-hardcoded-claim-mapper",
  "config": {
    "claim.name": "automatos:team",
    "claim.value": "ops",
    "jsonType.label": "String",
    "id.token.claim": "true",
    "access.token.claim": "false",
    "userinfo.token.claim": "false"
  }
}
```

The access rule `automatos:team=ops` admits its users. For access by group, add a mapper that puts the user's group membership in the ID token, and name that claim in the rule. Keycloak's mapper types are listed in the [Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/index.html); this page does not name one, because it has not been run here.

## 5. `auth_time`

Nothing to set. Keycloak's metadata lists `auth_time` among its claims, takes the `claims` parameter, and lists `none`, `login` and `consent` as prompt values (PRD-277 §5.9, read in Keycloak's source). The CI job checks a step-up end to end: Keycloak asks for the password again, and the step-up confirms.

## 6. The matching install settings

Chart values, with placeholders:

```yaml
existingSecret: automatos-secrets   # holds OIDC_CLIENT_SECRET
edition:
  authEdition: enterprise
  enterprise:
    publicUrl: https://automatos.example.com
    clientIpHeader: X-Real-IP
    bootstrapAdmins: email:admin@example.com
    oidc:
      issuer: https://id.example.com/realms/<realm>
      clientId: automatos
      displayName: Keycloak
      requiredClaims: automatos:team=ops
      caConfigMap: <configmap-name>   # only with a private certificate authority
      caKey: ca.crt
ingress:
  enabled: true
  host: automatos.example.com
```

The same as environment names:

```
AUTH_EDITION=enterprise
FRONTEND_URL=https://automatos.example.com
CLIENT_IP_HEADER=X-Real-IP
ENTERPRISE_BOOTSTRAP_ADMINS=email:admin@example.com
OIDC_ISSUER=https://id.example.com/realms/<realm>
OIDC_CLIENT_ID=automatos
OIDC_CLIENT_SECRET=<from the Kubernetes Secret>
OIDC_DISPLAY_NAME=Keycloak
OIDC_REQUIRED_CLAIMS=automatos:team=ops
OIDC_CA_BUNDLE_FILE=/etc/automatos/oidc-ca/ca.crt
```

The chart sets `OIDC_CA_BUNDLE_FILE` from `caConfigMap` and `caKey`. The access rule is optional for a realm that serves one organisation. It is required when `ENTERPRISE_DEFAULT_ROLE` is not `none`.

## 7. What this provider cannot do

Nothing Automatos needs was found missing. One difference to expect: Keycloak may ask the person to confirm sign-out, because Automatos sends no `id_token_hint` (see [Sign-out](#8-sign-out)).

## 8. Sign-out

- Keycloak's end-session address is `https://<keycloak-host>/realms/<realm>/protocol/openid-connect/logout`, on the issuer's host. Automatos sends the browser there with `client_id` and `post_logout_redirect_uri=https://automatos.example.com/sign-in`.
- Keycloak's guide says a request must "include either the `client_id` or `id_token_hint` parameter", and that `post_logout_redirect_uri` "needs to match one of the `Valid Post Logout Redirect URIs` specified in the client configuration".
- It also says "The user might be optionally requested to confirm the logout in case the `id_token_hint` parameter was not used".
- The CI job checks that Keycloak accepts the address Automatos sends.

## 9. Check it works

Follow the [checklist](enterprise-sign-in-oidc.md#check-a-registration-works). The CI job does the same against a throwaway realm. On your own realm, also check:

1. A user without the rule's claim is refused with `sign_in_not_allowed`.
2. A step-up after `STEP_UP_MAX_AGE_MINUTES` makes Keycloak ask for the password.
3. Sign-out shows Keycloak's confirm page, if it shows one, then returns to `/sign-in`.
