"""
Clerk JWT Verification for FastAPI
PRD-37: SaaS Foundation

Verifies Clerk JWTs using JWKS and extracts user/org claims.
"""

from typing import Optional, Dict, Any
import jwt
from jwt import PyJWKClient
import os
import logging
from functools import lru_cache
import httpx

logger = logging.getLogger(__name__)


class ClerkAuth:
    """
    Clerk JWT verification client.
    
    Verifies tokens issued by Clerk and extracts user/org claims.
    Uses JWKS (JSON Web Key Set) for secure signature verification.
    """
    
    def __init__(self):
        self.clerk_secret_key = os.getenv("CLERK_SECRET_KEY")
        self.clerk_publishable_key = os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")
        
        # Clerk JWKS URL - format: https://{clerk-frontend-api}/.well-known/jwks.json
        self.jwks_url = os.getenv("CLERK_JWKS_URL")
        self._jwks_client = None
        
        # Clerk API URL for fetching user details when email not in JWT
        # NOTE: This is a fallback. The proper fix is to configure Clerk Session Token Template
        # to include: {{user.primary_email_address}} in the email claim.
        # All Clerk instances use the same API endpoint
        self.clerk_api_base = "https://api.clerk.com"
        
        if not self.jwks_url:
            logger.warning(
                "CLERK_JWKS_URL not set. Auth will not work. "
                "Set this to your Clerk JWKS URL, e.g., "
                "https://your-app.clerk.accounts.dev/.well-known/jwks.json"
            )
        
        # Cache for email lookups to avoid repeated API calls
        self._email_cache: Dict[str, Optional[str]] = {}
    
    @property
    def jwks_client(self) -> Optional[PyJWKClient]:
        """Lazy-load the JWKS client."""
        if self._jwks_client is None and self.jwks_url:
            self._jwks_client = PyJWKClient(
                self.jwks_url,
                cache_jwk_set=True,
                lifespan=300  # Cache JWKS for 5 minutes
            )
        return self._jwks_client
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a Clerk JWT and return the claims.
        
        Args:
            token: The JWT token from Authorization header
            
        Returns:
            Dict with claims if valid, None if invalid or expired
        """
        if not self.jwks_client:
            logger.error("JWKS client not initialized. Cannot verify token.")
            return None
            
        try:
            # Get signing key from JWKS
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            
            # Verify and decode the token
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={
                    "verify_aud": False,  # Clerk doesn't use standard aud claim
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                }
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def extract_user_info(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract user information from JWT claims.
        
        Clerk tokens contain user data in various claim keys.
        This method normalizes them into a consistent format.
        
        Args:
            claims: Decoded JWT claims
            
        Returns:
            Normalized user info dict
        """
        # Extract system role from session token customization
        # Session token is configured to include: {"metadata": user.public_metadata}
        # We read from 'metadata' key first (new format), then fall back to 'public_metadata' (legacy)
        metadata = claims.get("metadata", {})
        if isinstance(metadata, dict):
            system_role = metadata.get("role")
        else:
            system_role = None
        
        # Fall back to legacy public_metadata or org_role
        if not system_role:
            public_metadata = claims.get("public_metadata", {})
            system_role = public_metadata.get("role") or claims.get("org_role") or "user"
        
        # Backwards-compatible alias to avoid NameError in older references
        role = system_role
        
        # Extract email - try multiple claim keys
        # Clerk JWT may have email in different places depending on configuration
        email = (
            claims.get("email") or 
            claims.get("primary_email") or
            claims.get("email_address") or
            claims.get("emailAddress")
        )
        
        # Some Clerk tokens include email_addresses + primary_email_address_id
        if not email:
            email_addresses = claims.get("email_addresses")
            primary_email_id = claims.get("primary_email_address_id")
            if isinstance(email_addresses, list):
                if primary_email_id:
                    for entry in email_addresses:
                        if isinstance(entry, dict) and entry.get("id") == primary_email_id:
                            email = entry.get("email_address")
                            break
                if not email:
                    for entry in email_addresses:
                        if isinstance(entry, dict) and entry.get("email_address"):
                            email = entry.get("email_address")
                            break
        
        # If still no email, try to fetch from Clerk API
        if not email:
            clerk_user_id = claims.get("sub")
            if clerk_user_id:
                # Check cache first
                if clerk_user_id in self._email_cache:
                    email = self._email_cache[clerk_user_id]
                else:
                    # Try to fetch from Clerk API
                    email = self._fetch_email_from_clerk_api(clerk_user_id)
                    self._email_cache[clerk_user_id] = email
        
        # If still no email after API fetch, generate a placeholder
        # This should never happen in production, but prevents crashes
        if not email:
            clerk_user_id = claims.get("sub")
            email = f"{clerk_user_id}@clerk.placeholder"
            # Only log once per user to reduce noise
            if clerk_user_id not in self._email_cache or self._email_cache[clerk_user_id] is None:
                logger.warning(
                    "⚠️  Clerk JWT missing email claim for user %s. Using placeholder: %s\n"
                    "🔧 TO FIX: Configure Clerk Session Token Template to include email:\n"
                    "   1. Go to Clerk Dashboard: https://dashboard.clerk.com\n"
                    "   2. Navigate to: Configure > Sessions > Customize session token\n"
                    "   3. Add this to your token template:\n"
                    '      { "email": "{{user.primary_email_address}}" }\n'
                    "   4. Save and users will need to sign in again for the change to take effect.\n"
                    "   (This is a configuration issue - users DO have emails, they're just not in the JWT)",
                    clerk_user_id,
                    email,
                )
                # Cache the placeholder to avoid repeated warnings
                self._email_cache[clerk_user_id] = email
        
        # Admin role is determined exclusively from Clerk publicMetadata / session token.
        # Domain-based auto-admin was removed for security (see PRD-43 US-025).
        
        return {
            "clerk_user_id": claims.get("sub"),
            "email": email,
            "name": claims.get("name") or claims.get("full_name"),
            "first_name": claims.get("first_name"),
            "last_name": claims.get("last_name"),
            "image_url": claims.get("image_url") or claims.get("profile_image_url"),
            # System role (from session token metadata or legacy public_metadata)
            "role": role,
            "system_role": system_role,  # Explicit key for system-level role
            # Organization claims (if user is in org context)
            "org_id": claims.get("org_id"),
            "org_role": claims.get("org_role"),
            "org_slug": claims.get("org_slug"),
            "org_permissions": claims.get("org_permissions", []),
        }
    
    def _fetch_email_from_clerk_api(self, clerk_user_id: str) -> Optional[str]:
        """
        Fetch user email from Clerk API when not present in JWT.
        
        This is a TEMPORARY WORKAROUND. The proper fix is to configure Clerk's
        Session Token Template to include: {{user.primary_email_address}}
        
        Users always have emails (required for signup), but they're not automatically
        included in JWTs - you must configure the Session Token Template in Clerk Dashboard.
        """
        if not self.clerk_secret_key:
            return None
        
        try:
            # Use Clerk Backend API to fetch user details
            # API endpoint: GET https://api.clerk.com/v1/users/{user_id}
            api_url = f"{self.clerk_api_base}/v1/users/{clerk_user_id}"
            headers = {
                "Authorization": f"Bearer {self.clerk_secret_key}",
                "Content-Type": "application/json",
            }
            
            # Use httpx with timeout to avoid hanging
            with httpx.Client(timeout=2.0) as client:
                response = client.get(api_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    # Extract primary email
                    email_addresses = data.get("email_addresses", [])
                    if email_addresses:
                        # Find primary email
                        primary = next(
                            (e for e in email_addresses if e.get("id") == data.get("primary_email_address_id")),
                            email_addresses[0]
                        )
                        email = primary.get("email_address")
                        if email:
                            logger.debug(f"✅ Fetched email from Clerk API (workaround) for user {clerk_user_id}: {email}")
                            logger.debug("   ⚠️  Consider configuring Session Token Template to include email in JWT")
                            return email
        except Exception as e:
            # Don't log errors for API fetch failures - it's just a fallback
            logger.debug(f"Failed to fetch email from Clerk API for user {clerk_user_id}: {e}")
        
        return None
    
    def is_configured(self) -> bool:
        """Check if Clerk auth is properly configured."""
        return bool(self.jwks_url and self.clerk_secret_key)

    async def get_org_members(self, org_id: str) -> list[dict]:
        """Fetch members of a Clerk organization."""
        if not self.clerk_secret_key:
            return []
        
        api_url = f"{self.clerk_api_base}/v1/organizations/{org_id}/memberships"
        headers = {
            "Authorization": f"Bearer {self.clerk_secret_key}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(api_url, headers=headers)
            if response.status_code == 200:
                return response.json().get("data", [])
            logger.error(f"Failed to fetch Clerk org members: {response.text}")
            return []

    async def invite_to_org(self, org_id: str, email: str, role: str, inviter_user_id: str) -> dict:
        """Invite a user to a Clerk organization."""
        if not self.clerk_secret_key:
            raise ValueError("Clerk secret key not configured")

        api_url = f"{self.clerk_api_base}/v1/organizations/{org_id}/invitations"
        headers = {
            "Authorization": f"Bearer {self.clerk_secret_key}",
            "Content-Type": "application/json",
        }
        
        # Clerk roles are org:admin, org:member usually. Map internal roles.
        clerk_role = "org:admin" if role in ["owner", "admin"] else "org:member"
        
        payload = {
            "email_address": email,
            "role": clerk_role,
            "inviter_user_id": inviter_user_id,
            "public_metadata": {"internal_role": role} # Store specific role in metadata
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()
            
            error_msg = response.json().get("errors", [{}])[0].get("message", "Unknown error")
            raise ValueError(f"Clerk Invitation Failed: {error_msg}")

    async def remove_from_org(self, org_id: str, user_id: str) -> None:
        """Remove a user from a Clerk organization."""
        if not self.clerk_secret_key:
            raise ValueError("Clerk secret key not configured")

        api_url = f"{self.clerk_api_base}/v1/organizations/{org_id}/memberships/{user_id}"
        headers = {
            "Authorization": f"Bearer {self.clerk_secret_key}",
            "Content-Type": "application/json",
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.delete(api_url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to remove user from Clerk org: {response.text}")
                # Don't raise error if they are arguably already gone, 
                # but good to log.



# Global instance for dependency injection
_clerk_auth: Optional[ClerkAuth] = None


def get_clerk_auth() -> ClerkAuth:
    """Get the global ClerkAuth instance."""
    global _clerk_auth
    if _clerk_auth is None:
        _clerk_auth = ClerkAuth()
    return _clerk_auth
