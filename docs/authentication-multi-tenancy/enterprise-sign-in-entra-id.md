# Enterprise Sign-In: Microsoft Entra ID

How to register Automatos with Microsoft Entra ID for the enterprise edition. Read [Enterprise Sign-In with OIDC](enterprise-sign-in-oidc.md) first: it explains the flow, the claims and every setting.

## 1. Evidence

- **Written from:** Microsoft's documentation, read on 21 November 2026, and this repository's code.
- **Not yet run against a live Entra ID tenant.** The conformance run (PRD-277 US-703) is that proof.
- Microsoft's pages used:
  - [Register an application](https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app) (updated 14 May 2026)
  - [How to add a redirect URI](https://learn.microsoft.com/en-us/entra/identity-platform/how-to-add-redirect-uri) (updated 14 May 2026)
  - [Add and manage application credentials](https://learn.microsoft.com/en-us/entra/identity-platform/how-to-add-credentials) (updated 12 November 2026)
  - [OpenID Connect on the Microsoft identity platform](https://learn.microsoft.com/en-us/entra/identity-platform/v2-protocols-oidc) (updated 6 October 2026)
  - [ID token claims reference](https://learn.microsoft.com/en-us/entra/identity-platform/id-token-claims-reference) (updated 15 June 2026)
  - [Optional claims reference](https://learn.microsoft.com/en-us/entra/identity-platform/optional-claims-reference) (updated 20 August 2026)
  - [Configure optional claims](https://learn.microsoft.com/en-us/entra/identity-platform/optional-claims) (updated 11 August 2026)
  - [Assign users and groups to an application](https://learn.microsoft.com/en-us/entra/identity/enterprise-apps/assign-user-or-group-access-portal) (updated 6 October 2026)

## 2. Register the application

In the Microsoft Entra admin center:

1. Go to **Entra ID** > **App registrations**.
2. Select **New registration**.
3. Enter a **Name**, for example `Automatos`.
4. Under **Supported account types**, select **Single tenant only - <your tenant>**.
5. Select **Register**.
6. On **Overview**, copy the **Application (client) ID**. This is `oidc.clientId`.
7. Note the tenant ID, a GUID. **App registrations** > your app > **Endpoints** shows it inside each address. You need it for the issuer.

Add the redirect addresses:

1. In the app registration, go to **Manage** > **Authentication**.
2. Open the **Redirect URI configuration** tab.
3. Select **Add Redirect URI**, then the **Web** platform.
4. Enter `https://automatos.example.com/api/v1/auth/oidc/callback`.
5. Select **Configure**.
6. Add a second **Web** redirect URI: `https://automatos.example.com/sign-in`. Microsoft requires the sign-out return address to be a registered redirect URI (see [Sign-out](#8-sign-out)).

Create the client secret:

1. Go to **Certificates & secrets** > **Client secrets**.
2. Select **New client secret**.
3. Enter a description and choose an expiry. Microsoft allows at most 24 months.
4. Select **Add**.
5. Copy the secret's **Value** now. Microsoft never shows it again.
6. Put it in the Kubernetes Secret that `existingSecret` names, under the key `OIDC_CLIENT_SECRET`.

Replace the secret before it expires: put the new value in the Kubernetes Secret and restart the API, which reads it at start.

Choose who may sign in. Two ways, which can be combined:

- **Assignment at Entra ID.** In **Enterprise apps** > **All applications** > your app > **Users and groups**, select **Add user/group**, choose people or groups, and select **Assign**. Microsoft's page says group-based assignment requires Microsoft Entra ID P1 or P2, and nested groups are not supported. Turn on the application's setting that requires assignment, so that only assigned people can sign in; Microsoft's page links to it under "Manage access to an application".
- **The access rule in Automatos** (`OIDC_REQUIRED_CLAIMS`), on `tid`, `roles` or `groups` (see [Claims](#4-claims)).

## 3. The issuer address

```
https://login.microsoftonline.com/<tenant-id>/v2.0
```

- Use the tenant ID, a GUID. Microsoft says the `iss` claim of a v2.0 token "ends in /v2.0" and identifies the tenant, and that "Your app should use the GUID portion". Automatos requires `iss` to equal `OIDC_ISSUER` exactly.
- Never use `common`, `organizations` or `consumers`. Those authorities serve many tenants. Automatos refuses to start with one of them unless an access rule is set.
- The metadata address is `https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration`. Entra shows it under **App registrations** > your app > **Endpoints**.
- `oidc.endpointHosts`: leave it empty. The token endpoint and the keys are on `login.microsoftonline.com`, the issuer's host (PRD-277 §5.9).

## 4. Claims

| Claim | What Microsoft says | Automatos setting |
|---|---|---|
| `sub` | "a pairwise identifier and is unique to an application ID". A new app registration gives everyone a new `sub`. | Do not use. Set `OIDC_CLAIM_SUBJECT=oid`. |
| `oid` | "The immutable identifier" for the person in the tenant; it "can't be reused". "the `profile` scope is required to receive this claim". | `OIDC_CLAIM_SUBJECT=oid`. Keep `profile` in `oidc.scopes` (it is in the default). |
| `tid` | The tenant's GUID. | Access rule: `tid=<tenant-id>`. |
| `email` | "isn't guaranteed to be correct and is mutable". Requested with the `email` scope. | Read, but see below. |
| `email_verified` | Not among the ID token claims Microsoft lists. | none |
| `name` | The person's display name. | `OIDC_CLAIM_NAME` default. |
| `roles` | "The set of roles that were assigned to the user" (app roles). | Access rule: `roles=<app-role-value>`. |
| `groups` | Optional. Added with **Token configuration** > **Add groups claim**. Above 200 groups in a JWT, Microsoft leaves the claim out and adds an overage claim instead. | Access rule: `groups=<value Entra sends for the group>`. |

Because Entra sends no `email_verified`, Automatos keeps no email address for Entra users. So:

- `email:` entries in `ENTERPRISE_BOOTSTRAP_ADMINS` never match. Use `sub:<object-id>`, the person's `oid` value.
- Automatos never matches or refuses a person by email address with Entra ID.

To add a groups claim: in the app registration, go to **Token configuration** > **Add groups claim**, and choose which groups. Microsoft recommends "Groups assigned to the application" for large organisations, because of the group limit.

## 5. `auth_time`

Entra ID sends `auth_time` only as an optional claim ("Time when the user last authenticated"). Add it:

1. In the app registration, go to **Manage** > **Token configuration**.
2. Select **Add optional claim**.
3. Choose the **ID** token type.
4. Select `auth_time`.
5. Select **Add**.

The manifest equivalent, from Microsoft's page:

```json
"optionalClaims": {
  "idToken": [{ "name": "auth_time", "essential": false }]
}
```

For a step-up and a phone's sign-in, Automatos sends `prompt=login` when Entra's metadata lists `login` among its prompt values or lists none. Microsoft says `prompt=login` "forces the user to enter their credentials". `max_age` is not among Microsoft's documented request parameters.

## 6. The matching install settings

Chart values, with placeholders:

```yaml
existingSecret: automatos-secrets   # holds OIDC_CLIENT_SECRET
edition:
  authEdition: enterprise
  enterprise:
    publicUrl: https://automatos.example.com
    clientIpHeader: X-Real-IP
    bootstrapAdmins: sub:<object-id-of-first-admin>
    oidc:
      issuer: https://login.microsoftonline.com/<tenant-id>/v2.0
      clientId: <application-client-id>
      displayName: Microsoft
      requiredClaims: tid=<tenant-id>
config:
  OIDC_CLAIM_SUBJECT: oid
ingress:
  enabled: true
  host: automatos.example.com
```

The same as environment names:

```
AUTH_EDITION=enterprise
FRONTEND_URL=https://automatos.example.com
CLIENT_IP_HEADER=X-Real-IP
ENTERPRISE_BOOTSTRAP_ADMINS=sub:<object-id-of-first-admin>
OIDC_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
OIDC_CLIENT_ID=<application-client-id>
OIDC_CLIENT_SECRET=<from the Kubernetes Secret>
OIDC_DISPLAY_NAME=Microsoft
OIDC_REQUIRED_CLAIMS=tid=<tenant-id>
OIDC_CLAIM_SUBJECT=oid
```

To admit only people with an app role, add it to the rule: `tid=<tenant-id>;roles=<app-role-value>`.

## 7. What this provider cannot do

- **No verified email.** Entra lists no `email_verified` claim, so Automatos keeps no address for its users, and `email:` first-admin entries do not match. Use `sub:` entries.
- **Large group lists are left out.** Above 200 groups, the `groups` claim is not in the token. A rule on `groups` refuses such a person with `sign_in_not_allowed`. A rule on `roles` is not affected.
- **`sub` changes with the app registration.** Use `oid`, which is the same for every app in the tenant.
- **`auth_time` is off by default.** Without the optional claim, every step-up is refused with `step_up_no_auth_time`, and every phone sign-in with `sign_in_refused`.

## 8. Sign-out

- Entra's metadata lists an `end_session_endpoint`, `https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/logout`, on the issuer's host. Automatos sends the browser there with `client_id` and `post_logout_redirect_uri=https://automatos.example.com/sign-in`.
- Microsoft says `post_logout_redirect_uri` "must match one of the redirect URIs registered for your application". So `https://automatos.example.com/sign-in` must be a registered Web redirect URI.
- Automatos sends no `logout_hint`. Microsoft documents `logout_hint` as the way to sign out "without prompting the user to select an account", so Entra may ask the person to choose an account.
- Microsoft says the platform "ends the user session for this application. However, the user may still be signed in to other applications".

## 9. Check it works

Follow the [checklist](enterprise-sign-in-oidc.md#check-a-registration-works), and check these Entra-specific points:

1. Sign in as the first admin. You become an install admin, which shows the `sub:` entry holds the right `oid`.
2. Sign in as a person who is not assigned to the app. Entra refuses them, if assignment is required.
3. Change a person's role after `STEP_UP_MAX_AGE_MINUTES`. Entra asks for your credentials, and the change goes through. `step_up_no_auth_time` means the `auth_time` optional claim is missing.
4. Sign out. Entra's sign-out page appears, then you return to `/sign-in`.
