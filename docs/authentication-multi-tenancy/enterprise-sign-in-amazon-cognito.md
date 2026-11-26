# Enterprise Sign-In: Amazon Cognito

How to register Automatos with an Amazon Cognito user pool for the enterprise edition. Read [Enterprise Sign-In with OIDC](enterprise-sign-in-oidc.md) first: it explains the flow, the claims and every setting.

## 1. Evidence

- **Written from:** Amazon's documentation, read on 21 November 2026, and this repository's code.
- **Not yet run against a live user pool.** The conformance run (PRD-277 US-703) is that proof.
- Amazon's pages used (Amazon Cognito Developer Guide):
  - [Application-specific settings with app clients](https://docs.aws.amazon.com/cognito/latest/developerguide/user-pool-settings-client-apps.html)
  - [User pool managed login](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-managed-login.html)
  - [User pool endpoints and managed login reference](https://docs.aws.amazon.com/cognito/latest/developerguide/federation-endpoints.html)
  - [The authorize endpoint](https://docs.aws.amazon.com/cognito/latest/developerguide/authorization-endpoint.html)
  - [The logout endpoint](https://docs.aws.amazon.com/cognito/latest/developerguide/logout-endpoint.html)
  - [Understanding the identity (ID) token](https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-the-id-token.html)
  - [Quotas in Amazon Cognito](https://docs.aws.amazon.com/cognito/latest/developerguide/quotas.html)

## 2. Register the application

The user pool needs a domain. Amazon says that for the authorization code grant "you must add a domain to your user pool".

Set the domain's branding version to managed login:

1. Open the user pool's **Domain** tab.
2. Set the branding version to **Managed login**. The other choice, **Hosted UI (classic)**, applies to all app clients too.

Amazon says: "In the Lite feature plan, the classic hosted UI is your only option." A step-up needs managed login (see [`auth_time`](#5-auth_time)).

Create the app client in the Amazon Cognito console:

1. Open **User pools** and choose your pool.
2. Create an app client. Choose the application type **Traditional web application**. Amazon says this type generates a client secret, and "You can't change whether an app client has a client secret after you create the app client."
3. Enter a name.
4. Enter the return URL `https://automatos.example.com/api/v1/auth/oidc/callback`.
5. Select **Create app client**.

Then edit the app client's settings:

1. **Allowed callback URLs:** `https://automatos.example.com/api/v1/auth/oidc/callback`.
2. **Allowed sign-out URLs:** `https://automatos.example.com/sign-in`.
3. **OAuth grant types:** **Authorization code grant**.
4. **OpenID Connect scopes:** `openid`, `email` and `profile`.

Copy the client ID and the client secret. Put the secret in the Kubernetes Secret that `existingSecret` names, under the key `OIDC_CLIENT_SECRET`.

Amazon's app client page describes an option to have Amazon Cognito manage the secret in AWS Secrets Manager, where "Amazon Cognito stores and rotates the secret". Automatos reads `OIDC_CLIENT_SECRET` from the Kubernetes Secret at start. Keep that value the same as the app client's current secret, and restart the API after changing it.

## 3. The issuer address

```
https://cognito-idp.<region>.amazonaws.com/<user-pool-id>
```

- Amazon's endpoints page names two issuer forms: the original `https://cognito-idp.<region>.amazonaws.com/<user-pool-id>` and the updated `https://issuer-cognito-idp.<region>.amazonaws.com/<user-pool-id>`.
- Automatos reads `<issuer>/.well-known/openid-configuration` and requires its `issuer` to equal `oidc.issuer` exactly. Open that address for your pool and copy the `issuer` value it shows.
- The keys (`jwks.json`) are on the issuer's host. The token endpoint is on the user pool's domain: `https://<prefix>.auth.<region>.amazoncognito.com/oauth2/token`, or your custom domain. List that domain in `oidc.endpointHosts`.

## 4. Claims

| Claim | What Amazon says | Automatos setting |
|---|---|---|
| `sub` | "the best way to identify a given user". | `OIDC_CLAIM_SUBJECT` default (`sub`). |
| `email`, `email_verified` | Sent with the `email` scope. | Defaults. The address is kept when `email_verified` is `true`. |
| `cognito:groups` | An array of the user pool groups the user belongs to. | Access rule: `cognito:groups=<group-name>`. |
| `auth_time` | Listed among the ID token's claims. | See [`auth_time`](#5-auth_time). |

Amazon says the OIDC claims `iss`, `sub`, `aud`, `exp` and `iat` "are present in every Amazon Cognito ID token", and to "Treat every other claim as optional."

The access rule reads `cognito:groups` by its literal name, colon included. A user can belong to at most 100 groups (Amazon's quotas page).

## 5. `auth_time`

Amazon's example ID token from the authorization code flow carries `auth_time`. Amazon does not list it among the claims present in every token.

For a step-up and a phone's sign-in, Automatos sends `max_age=0`, and `prompt=login` when Cognito's metadata lists `login` or lists no prompt values. What Amazon says about each:

- `prompt` is "Available in the managed login branding version only, not in the classic hosted UI". `prompt=login` "requires users to re-authenticate even if they have an existing session".
- With managed login, "If the current time minus auth_time is greater than max_age, the user must authenticate again from scratch".
- With the classic hosted UI, Cognito "silently ignores acr_values and max_age".

So use managed login. With the classic hosted UI, a step-up confirms only when Cognito returns an `auth_time` no earlier than the step-up's start; otherwise it is refused with `step_up_refused`.

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
      issuer: https://cognito-idp.<region>.amazonaws.com/<user-pool-id>
      clientId: <app-client-id>
      displayName: Amazon Cognito
      requiredClaims: cognito:groups=<group-name>
      endpointHosts: <prefix>.auth.<region>.amazoncognito.com
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
OIDC_ISSUER=https://cognito-idp.<region>.amazonaws.com/<user-pool-id>
OIDC_CLIENT_ID=<app-client-id>
OIDC_CLIENT_SECRET=<from the Kubernetes Secret>
OIDC_DISPLAY_NAME=Amazon Cognito
OIDC_REQUIRED_CLAIMS=cognito:groups=<group-name>
OIDC_ENDPOINT_HOSTS=<prefix>.auth.<region>.amazoncognito.com
DEFAULT_WORKSPACE_ID=00000000-0000-0000-0000-0000000000c1
DATABASE_SSLMODE=verify-full
DATABASE_SSLROOTCERT=/etc/automatos/database-ca/ca.crt
```

The chart sets the last three itself: `DEFAULT_WORKSPACE_ID` from `edition.defaultWorkspaceId`, and `DATABASE_SSLMODE` and `DATABASE_SSLROOTCERT` from `database.*`. Set them yourself only for an install made without the chart. The enterprise edition connects to its database only with `verify-full` ([chart README, Settings](../../charts/automatos/README.md#settings)).

The access rule is optional for a pool that serves one organisation. It is required when `ENTERPRISE_DEFAULT_ROLE` is not `none`.

## 7. What this provider cannot do

- **No step-up prompt with the classic hosted UI.** `prompt` and `max_age` work only with managed login, and the Lite feature plan has only the classic hosted UI.
- **`auth_time` is not guaranteed.** Amazon treats every claim beyond `iss`, `sub`, `aud`, `exp` and `iat` as optional. A token without it refuses each step-up with `step_up_no_auth_time`.
- **Groups per user.** At most 100.
- **Sign-out takes different parameters.** See [Sign-out](#8-sign-out).

## 8. Sign-out

- Cognito's `/logout` endpoint is on the user pool's domain. Amazon says it takes `client_id` with `logout_uri`, which must be one of the **Allowed sign-out URLs**, or `redirect_uri`: "Amazon Cognito requires either a logout_uri or a redirect_uri parameter".
- The browser is sent to the provider only when its metadata lists an `end_session_endpoint`. The pages read do not say whether Cognito's metadata lists one.
- If the pool's metadata lists an `end_session_endpoint`, Cognito may refuse the request, which carries `client_id` and `post_logout_redirect_uri` where Cognito's logout endpoint asks for `logout_uri` or `redirect_uri`. The Automatos session has ended either way.
- When the metadata lists none, sign-out ends the Automatos session, and Cognito's managed login session continues. Amazon says that session cookie lasts one hour.
- Amazon also says: "The logout endpoint doesn't sign users out of OIDC or social identity providers".
- The conformance run (US-703) records what happens.

## 9. Check it works

Follow the [checklist](enterprise-sign-in-oidc.md#check-a-registration-works), and check these Cognito-specific points:

1. Sign in as a user in the rule's group. You reach Automatos.
2. Sign in as a user outside it. You see `sign_in_not_allowed`.
3. Change a person's role after `STEP_UP_MAX_AGE_MINUTES`. Cognito's managed login asks you to sign in again, and the change goes through. `step_up_no_auth_time` means the token has no `auth_time`.
4. Sign out. Note whether the browser goes to Cognito, and where it lands.
