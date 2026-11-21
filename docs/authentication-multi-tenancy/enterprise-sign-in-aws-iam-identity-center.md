# Enterprise Sign-In: AWS IAM Identity Center

AWS IAM Identity Center is **not supported** as the enterprise edition's OIDC provider. This page gives AWS's reason, in AWS's words, and the path AWS documents instead. Read [Enterprise Sign-In with OIDC](enterprise-sign-in-oidc.md) for how sign-in works.

## Evidence

- **Written from:** AWS's documentation, read on 21 November 2026.
- **The alternative below is documented by AWS and not yet run here.** No IAM Identity Center or Cognito federation has been run against an install.
- AWS's pages used:
  - [AWS IAM Identity Center FAQs](https://aws.amazon.com/iam/identity-center/faqs/)
  - [Customer managed applications](https://docs.aws.amazon.com/singlesignon/latest/userguide/customermanagedapps.html)
  - [SAML 2.0 and OAuth 2.0 applications](https://docs.aws.amazon.com/singlesignon/latest/userguide/customermanagedapps-saml2-oauth2.html)
  - [Using applications with a trusted token issuer](https://docs.aws.amazon.com/singlesignon/latest/userguide/using-apps-with-trusted-token-issuer.html)
  - [Attribute mappings](https://docs.aws.amazon.com/singlesignon/latest/userguide/attributemappingsconcept.html)
  - [SAML identity providers in a user pool](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-saml-idp.html) (Amazon Cognito Developer Guide)
  - [Setting up IAM Identity Center with a Cognito user pool](https://docs.aws.amazon.com/res/latest/ug/sso-idc.html) (Research and Engineering Studio user guide)

## Why it is not supported

Automatos signs people in as an OpenID Connect relying party. AWS says IAM Identity Center does not offer that to an application like Automatos:

- The FAQ asks: "My application supports OpenID Connect (OIDC) only. Can I use it with IAM Identity Center?" It answers: "No. IAM Identity Center supports only SAML 2.0–based applications."
- The customer managed applications page says: "If your application supports SAML 2.0, you can connect it to an organization instance of IAM Identity Center."
- IAM Identity Center's OAuth 2.0 support is for AWS services. The SAML 2.0 and OAuth 2.0 applications page says: "OAuth 2.0 grant flows are only available through AWS managed applications that support the flows", and "These grant types can be used only with AWS services that support this capability."
- A trusted token issuer works the other way round. AWS says it "is an OAuth 2.0 authorization server that creates signed tokens", whose tokens are exchanged for IAM Identity Center tokens because "AWS services cannot use the tokens from the external token issuer directly".

## The documented alternative: a Cognito user pool

AWS documents connecting IAM Identity Center to an Amazon Cognito user pool over SAML 2.0. The install then signs in against Cognito, which is an OIDC provider. **Documented by AWS; not yet run here.**

1. Set up the user pool and its app client for Automatos, as in [Amazon Cognito](enterprise-sign-in-amazon-cognito.md).
2. In IAM Identity Center, add a SAML 2.0 application. AWS's worked example: **Applications** > **Add application** > **I have an application I want to set up** > **SAML 2.0**.
3. Copy the **IAM Identity Center SAML metadata** address it shows.
4. Set the application's ACS URL to `https://<user-pool-domain>/saml2/idpresponse`.
5. Set the application's SAML audience to `urn:amazon:cognito:sp:<user-pool-id>`.
6. Map the attributes. AWS's worked example maps **Subject** to `${user:email}` (format `emailAddress`) and adds `email`. Cognito's SAML page says to "Map NameId … from an IdP attribute that has values that don't change". IAM Identity Center's attribute list includes `${user:subject}`.
7. Assign the users or groups who may use Automatos.
8. In the Cognito user pool, add a SAML identity provider from the metadata address in step 3.
9. Assign that identity provider to the Automatos app client.

The install's settings are the Cognito ones: `oidc.issuer` is the user pool's issuer, and `oidc.endpointHosts` its domain.

Not confirmed on the pages read:

- which `email_verified` value Cognito puts in the ID token of a user who signed in through a SAML provider. Without `true`, Automatos keeps no address and `email:` first-admin entries do not match; use `sub:` entries;
- whether IAM Identity Center group membership can reach `cognito:groups`.
