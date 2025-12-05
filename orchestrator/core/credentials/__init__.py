"""
Credentials - Secure Credential Management
==========================================

Core infrastructure for managing API keys, database credentials,
and other secrets securely.

Components:
- service.py    - CredentialStore for CRUD operations
- resolver.py   - CredentialResolver for runtime resolution
- tester.py     - CredentialTester for validation
- encryption.py - Encryption/decryption services

Usage:
    from core.credentials import CredentialStore, get_credential_resolver
    
    store = CredentialStore(db)
    resolver = get_credential_resolver()
"""

from core.credentials.service import (
    CredentialStore,
    CredentialCreate,
    CredentialUpdate,
)
from core.credentials.resolver import (
    CredentialResolver,
    get_credential_resolver,
)
from core.credentials.tester import (
    CredentialTester,
)
from core.credentials.encryption import (
    EncryptionService,
    get_encryption_service,
    EncryptionKeyError,
)

__all__ = [
    # Service
    'CredentialStore',
    'CredentialCreate',
    'CredentialUpdate',
    
    # Resolver
    'CredentialResolver',
    'get_credential_resolver',
    
    # Tester
    'CredentialTester',
    
    # Encryption
    'EncryptionService',
    'get_encryption_service',
    'EncryptionKeyError',
]

