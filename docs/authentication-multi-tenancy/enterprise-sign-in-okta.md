# Enterprise Sign-In: Okta

How to register Automatos with Okta for the enterprise edition. Read [Enterprise Sign-In with OIDC](enterprise-sign-in-oidc.md) first: it explains the flow, the claims and every setting.

## 1. Evidence

- **Written from:** Okta's documentation, read on 21 November 2026, and this repository's code.
- **Not yet run against a live Okta org.** The conformance run (PRD-277 US-703) is that proof.
- Okta's pages used:
  - [Create OIDC app integrations](https://help.okta.com/en-us/content/topics/apps/apps_app_integration_wizard_oidc.htm)
  - [Authorization servers](https://developer.okta.com/docs/concepts/auth-servers/)
  - [Customize tokens with a groups claim](https://developer.okta.com/docs/guides/customize-tokens-groups-claim/main/)
  - [OpenID Connect and OAuth 2.0 API overview](https://developer.okta.com/docs/api/openapi/okta-oauth/guides/overview) (last updated 6 October 2026)
  - [Authorize](https://developer.okta.com/docs/api/openapi/okta-oauth/oauth/orgas/authorize) and [Logout](https://developer.okta.com/docs/api/openapi/okta-oauth/oauth/orgas/logout) (API reference)

## 2. Register the application

In the Okta Admin Console:

1. Go to **Applications and Resources** > **Applications**.
2. Select **Create App Integration**. Okta's page describes the **Classic experience** wizard.
3. Choose **OIDC - OpenID Connect** as the sign-in method.
4. Choose **Web Application** as the application type. Select **Next**.
5. Enter a name.
6. Under the grant type, keep **Authorization code**.
7. Under **Sign-in redirect URIs**, enter `https://automatos.example.com/api/v1/auth/oidc/callback`.
8. Under **Sign-out redirect URIs**, enter `https://automatos.example.com/sign-in`.
9. Under **Assignments**, choose **Controlled access** ("Limit access to selected groups") and pick the groups that may use Automatos.
10. Select **Save**. Okta generates the client secret.
11. Copy the client ID. This is `oidc.clientId`.
12. Copy the client secret. Put it in the Kubernetes Secret that `existingSecret` names, under the key `OIDC_CLIENT_SECRET`.

Okta's page says PKCE is optional for a web application. Automatos sends PKCE on every sign-in whatever the app integration's setting.

## 3. The issuer address

Okta has two kinds of authorization server. Choose one:

| Server | Issuer | Notes |
|---|---|---|
| Org authorization server | `https://<okta-domain>` | Built in. Okta: "You can add custom claims to ID tokens, but custom scopes aren't supported". |
| Custom authorization server | `https://<okta-domain>/oauth2/<server-id>`, for example `/oauth2/default` | Needs API Access Management, which Okta calls an "optional add-on". |

- The metadata address is `<issuer>/.well-known/openid-configuration`.
- `oidc.endpointHosts`: leave it empty. The token endpoint and the keys are on the Okta domain, the issuer's host (PRD-277 §5.9).
- Use the same server for the app's groups claim (below).

## 4. Claims

| Claim | What Okta says | Automatos setting |
|---|---|---|
| `sub` | One of the base claims, always present. | `OIDC_CLAIM_SUBJECT` default (`sub`). |
| `email`, `email_verified` | Sent with the `email` scope. | Defaults. The address is kept when `email_verified` is `true`. |
| `name` | A profile claim. | `OIDC_CLAIM_NAME` default. |
| `auth_time` | A base claim, always present. | Nothing to set. |
| `groups` | Only once a groups claim is configured. | Access rule: `groups=<group-name>`. |

Add a groups claim to the ID token:

- **Org authorization server.** In the app integration, go to the **Sign On** tab and select **Edit** in the **OpenID Connect ID Token** section. Set the group claim type to **Filter**, keep the claim name `groups`, and choose a filter. Okta's example is **Matches regex** with `.*`. Save. The org server includes the claim "when you include the `groups` scope in the request", so set `oidc.scopes: openid profile email groups`.
- **Custom authorization server.** Under **Security** > **API**, open the server and go to the **Claims** tab. Select **Add Claim**. Name it `groups`. Under **Include in token type**, choose the ID token; Okta's example keeps the access token. Set **Value type** to **Groups**, and choose a filter.

A filter that matches only the groups the access rule names keeps the list short. Okta says "the groups claim has a limit of 100. If more than 100 groups match the filter, then the request fails."

## 5. `auth_time`

Nothing to set. Okta lists `auth_time` among the "base claims (always present)".

For a step-up and a phone's sign-in, Automatos sends `max_age=0`, and `prompt=login` when Okta's metadata lists `login` among its prompt values or lists none. Okta documents both parameters. For `prompt=login` Okta says: "Always prompt the user for authentication, regardless of whether they have an Okta session."

## 6. The matching install settings

Chart values for the org authorization server with a groups claim, with placeholders:

```yaml
existingSecret: automatos-secrets   # holds OIDC_CLIENT_SECRET
edition:
  authEdition: enterprise
  enterprise:
    publicUrl: https://automatos.example.com
    clientIpHeader: X-Real-IP
    bootstrapAdmins: email:admin@example.com
    oidc:
      issuer: https://<okta-domain>
      clientId: <client-id>
      displayName: Okta
      scopes: openid profile email groups
      requiredClaims: groups=<group-name>
database:
  sslmode: verify-full
  caConfigMap: <database-ca-configmap>   # the CA that signed the database server's certificate
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
OIDC_ISSUER=https://<okta-domain>
OIDC_CLIENT_ID=<client-id>
OIDC_CLIENT_SECRET=<from the Kubernetes Secret>
OIDC_DISPLAY_NAME=Okta
OIDC_SCOPES=openid profile email groups
OIDC_REQUIRED_CLAIMS=groups=<group-name>
DEFAULT_WORKSPACE_ID=00000000-0000-0000-0000-0000000000c1
DATABASE_SSLMODE=verify-full
DATABASE_SSLROOTCERT=/etc/automatos/database-ca/ca.crt
```

The chart sets the last three itself: `DEFAULT_WORKSPACE_ID` from `edition.defaultWorkspaceId`, and `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT` from `database.*`. Set them yourself only for an install made without the chart. The enterprise edition connects to its database only with `verify-full` ([chart README, Settings](../../charts/automatos/README.md#settings)).

The access rule is optional for an Okta org, which serves one organisation. It is required when `ENTERPRISE_DEFAULT_ROLE` is not `none`. Without a groups claim, drop `groups` from the scopes and the rule, and control access with the app's **Assignments**.

## 7. What this provider cannot do

- **More than 100 matching groups fails the sign-in at Okta.** Keep the groups filter narrow.
- **No custom scopes on the org server.** Custom claims still work there.
- Okta's logout reference lists `id_token_hint` as required. The sign-out request carries `client_id` and `post_logout_redirect_uri` (see [Sign-out](#8-sign-out)).

## 8. Sign-out

- Automatos sends the browser to Okta's `end_session_endpoint`, when the metadata lists one on the Okta domain, with `client_id` and `post_logout_redirect_uri=https://automatos.example.com/sign-in`.
- Okta's logout reference lists `id_token_hint` as required, and says `post_logout_redirect_uri` "must match the value preregistered". Register `/sign-in` as a sign-out redirect URI.
- What Okta does with a request without `id_token_hint` is not covered by the pages read. The person may stay signed in at Okta. The conformance run (US-703) records it.
- The Automatos session has ended either way.

## 9. Check it works

Follow the [checklist](enterprise-sign-in-oidc.md#check-a-registration-works), and check these Okta-specific points:

1. Sign in as a person in an assigned group. You reach Automatos.
2. Sign in as a person outside every assigned group. Okta refuses them.
3. With a groups rule, sign in as an assigned person outside the rule's group. You see `sign_in_not_allowed`.
4. Change a person's role after `STEP_UP_MAX_AGE_MINUTES`. Okta asks you to sign in again, and the change goes through.
5. Sign out. Note whether Okta's page appears and whether you return to `/sign-in`.
