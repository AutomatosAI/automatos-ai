# Enterprise Sign-In: Google Workspace

How to register Automatos with Google for the enterprise edition, so people sign in with their Google Workspace accounts. Read [Enterprise Sign-In with OIDC](enterprise-sign-in-oidc.md) first: it explains the flow, the claims and every setting.

## 1. Evidence

- **Written from:** Google's documentation and Google's published metadata document, read on 21 November 2026, and this repository's code.
- **Not yet run against a live Google Workspace domain.** The conformance run (PRD-277 US-703) is that proof.
- Google's pages used (none shows an update date):
  - [OpenID Connect](https://developers.google.com/identity/openid-connect/openid-connect)
  - [Google's metadata document](https://accounts.google.com/.well-known/openid-configuration)
  - [Security bundle: session age claims](https://developers.google.com/identity/siwg/security-bundle)
  - [Manage OAuth clients](https://support.google.com/cloud/answer/15549257)
  - [Manage app audience](https://support.google.com/cloud/answer/15549945)

## 2. Register the application

In the Google Cloud console, in the project that will hold the client:

1. Open **Google Auth Platform**.
2. Under **Audience**, choose who may use the app. Google says **Internal** limits authorization requests "to members of the organization"; **External** admits any Google Account.
3. Open **Clients** and select **CREATE CLIENT**.
4. Choose the web application type and enter a name.
5. Under the authorized redirect URIs, add `https://automatos.example.com/api/v1/auth/oidc/callback`. Google says a web application "must specify authorized redirect URIs".
6. Select **CREATE**.
7. Copy the client ID. This is `oidc.clientId`.
8. Copy or download the client secret now. Google shows it only at creation.
9. Put the secret in the Kubernetes Secret that `existingSecret` names, under the key `OIDC_CLIENT_SECRET`.

No sign-out address is registered: Google publishes no end-session endpoint (see [Sign-out](#8-sign-out)).

## 3. The issuer address

```
https://accounts.google.com
```

- Google says its ID tokens' `iss` is "Always https://accounts.google.com or accounts.google.com". Automatos takes only the https form.
- This issuer serves every Google account. Automatos refuses to start with it unless an access rule is set. Use `hd=<your-domain>`.
- The token endpoint is `https://oauth2.googleapis.com/token` and the keys are at `https://www.googleapis.com/oauth2/v3/certs`. Both are off the issuer's host, so set `oidc.endpointHosts: oauth2.googleapis.com,www.googleapis.com`.
- Google's metadata offers `client_secret_basic`, which Automatos uses, and PKCE `S256`.

## 4. Claims

| Claim | What Google says | Automatos setting |
|---|---|---|
| `sub` | "unique among all Google Accounts and never reused". | `OIDC_CLAIM_SUBJECT` default (`sub`). |
| `email` | Sent with the `email` scope. | `OIDC_CLAIM_EMAIL` default. |
| `email_verified` | Sent with `email`. | Read: the address is kept when it is `true`. |
| `name` | Listed in the metadata's `claims_supported`. | `OIDC_CLAIM_NAME` default. |
| `hd` | "Provided only if the user belongs to a Google Cloud organization." | Access rule: `hd=<your-domain>`. |
| `auth_time` | "Provided when the auth_time claim is included in the authentication request and enabled in settings." | See [`auth_time`](#5-auth_time). |

Google's metadata lists no groups claim. The default scopes, `openid profile email`, are enough.

Because Google sends a verified `email`, `email:` entries in `ENTERPRISE_BOOTSTRAP_ADMINS` work.

## 5. `auth_time`

Two things are needed.

1. **The request.** Automatos sends the `claims` parameter asking for `auth_time` on every sign-in. Google documents it as the way to ask for `auth_time`. Nothing to set.
2. **The project setting.** Google's security bundle page gives these steps:
   1. Open **Google Auth Platform** for the project.
   2. Under **Audience**, confirm "*Publishing status* is *In production*".
   3. Under **Verification Center**, confirm "*Verification Status* is *Verified*".
   4. Open **Settings**.
   5. Under **Advanced Settings**, select **Session age claims** to enable `auth_time`.

Google says: "To receive additional claims your app needs to be published, verified, and security bundle features enabled." This is a setting of the project in Google Auth Platform, not of the OAuth client.

Google's audience page says the publishing status "is not applicable to Internal or Personal projects". The pages read do not say whether an Internal app can turn on session age claims. Check the **Settings** page of your project.

For a step-up and a phone's sign-in, the request carries `max_age=0` and no `prompt=login`: Google lists only `none`, `consent` and `select_account` as `prompt` values. Google's documentation lists no `max_age` parameter.

So a step-up is confirmed only when the person's Google session began after the step-up started. A person who is still signed in to Google is refused with `step_up_refused`. What to do: sign out of Google, for example from the account menu in another Google tab, then make the change in Automatos again. The new step-up sends you to Google, Google asks you to sign in, and its `auth_time` is after the step-up's start. Not yet run against a live domain.

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
      issuer: https://accounts.google.com
      clientId: <client-id>.apps.googleusercontent.com
      displayName: Google
      requiredClaims: hd=example.com
      endpointHosts: oauth2.googleapis.com,www.googleapis.com
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
OIDC_ISSUER=https://accounts.google.com
OIDC_CLIENT_ID=<client-id>.apps.googleusercontent.com
OIDC_CLIENT_SECRET=<from the Kubernetes Secret>
OIDC_DISPLAY_NAME=Google
OIDC_REQUIRED_CLAIMS=hd=example.com
OIDC_ENDPOINT_HOSTS=oauth2.googleapis.com,www.googleapis.com
DEFAULT_WORKSPACE_ID=00000000-0000-0000-0000-0000000000c1
DATABASE_SSLMODE=verify-full
DATABASE_SSLROOTCERT=/etc/automatos/database-ca/ca.crt
```

The chart sets the last three itself: `DEFAULT_WORKSPACE_ID` from `edition.defaultWorkspaceId`, and `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT` from `database.*`. Set them yourself only for an install made without the chart. The enterprise edition connects to its database only with `verify-full` ([chart README, Settings](../../charts/automatos/README.md#settings)).

`hd=example.com` is required: Automatos refuses to start with Google's issuer and no access rule.

## 7. What this provider cannot do

- **No groups.** Google sends no groups, so access by group is not available with Google. Access is by domain (`hd`), and by the app's **Audience** at Google.
- **No end-session endpoint.** Google publishes none, so sign-out ends the Automatos session; the Google session continues.
- **No documented way to ask for the password again.** Google lists no `prompt=login` and no `max_age`. A step-up confirms only when Google returns an `auth_time` no earlier than the step-up's start. When Google returns an earlier sign-in time, the step-up is refused with `step_up_refused`: sign out of Google and try again (see [`auth_time`](#5-auth_time)). A phone's sign-in follows the same rule, and is refused with `sign_in_refused`; such a phone pairs with a code from Settings.
- **`auth_time` needs a published, verified project** with session age claims on. Without it, every step-up is refused with `step_up_no_auth_time`.

## 8. Sign-out

Google publishes no end-session endpoint, so sign-out ends the Automatos session; the Google session continues. Google's metadata document lists no `end_session_endpoint`, and `POST /api/v1/auth/sign-out` answers `204`.

## 9. Check it works

Follow the [checklist](enterprise-sign-in-oidc.md#check-a-registration-works), and check these Google-specific points:

1. Sign in with an account in your domain. You reach Automatos.
2. Sign in with a Google account outside your domain. You see `sign_in_not_allowed`, or Google refuses it if the audience is Internal.
3. Sign in as the `email:` first admin. You become an install admin, which shows Google sent `email_verified`.
4. Change a person's role after `STEP_UP_MAX_AGE_MINUTES`. Note whether Google asks for your password, and whether the change goes through, or is refused with `step_up_refused` or `step_up_no_auth_time`.
5. Sign out. You land on `/sign-in`, and your Google account stays signed in.
