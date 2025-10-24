"""
Encryption Service for Credential Management
============================================

Handles encryption/decryption of sensitive credentials using Fernet symmetric encryption.
Auto-generates encryption key on first run with secure storage.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class EncryptionKeyError(Exception):
    """Raised when encryption key operations fail"""
    pass


class EncryptionService:
    """
    Manages encryption/decryption of sensitive data using Fernet symmetric encryption.
    
    Key Management Strategy:
    1. Try environment variable CREDENTIAL_ENCRYPTION_KEY
    2. Try .credential_key file in orchestrator directory
    3. Auto-generate new key and save to .credential_key
    
    Security Features:
    - Fernet (AES-128-CBC with HMAC-SHA256)
    - Base64 encoded encrypted values for database storage
    - Automatic key rotation support (future)
    - Key file permissions enforced (0o600)
    """
    
    def __init__(self):
        """Initialize encryption service and load/generate encryption key"""
        self.key = self._load_or_generate_key()
        self.cipher = Fernet(self.key)
        logger.info("Encryption service initialized")
    
    def _load_or_generate_key(self) -> bytes:
        """
        Load encryption key from environment or file, or generate new one.
        
        Priority Order:
        1. CREDENTIAL_ENCRYPTION_KEY environment variable
        2. .credential_key file
        3. Generate new key
        
        Returns:
            bytes: Fernet encryption key
            
        Raises:
            EncryptionKeyError: If key loading/generation fails
        """
        # Try environment variable first
        key_str = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
        if key_str:
            logger.info("Using encryption key from environment variable")
            try:
                # Validate it's a valid Fernet key
                key_bytes = key_str.encode() if isinstance(key_str, str) else key_str
                Fernet(key_bytes)  # Test key validity
                return key_bytes
            except Exception as e:
                logger.error(f"Invalid encryption key in environment variable: {e}")
                raise EncryptionKeyError("Invalid CREDENTIAL_ENCRYPTION_KEY in environment")
        
        # Try key file
        key_file = Path(__file__).parent.parent / ".credential_key"
        
        if key_file.exists():
            logger.info(f"Loading encryption key from {key_file}")
            try:
                key_bytes = key_file.read_bytes().strip()
                Fernet(key_bytes)  # Test key validity
                return key_bytes
            except Exception as e:
                logger.error(f"Invalid encryption key in file {key_file}: {e}")
                raise EncryptionKeyError(f"Invalid encryption key in {key_file}")
        
        # Generate new key
        logger.warning("No encryption key found - generating new key")
        return self._generate_and_save_key(key_file)
    
    def _generate_and_save_key(self, key_file: Path) -> bytes:
        """
        Generate new encryption key and save to file.
        
        Args:
            key_file: Path to save the key
            
        Returns:
            bytes: Generated Fernet key
            
        Raises:
            EncryptionKeyError: If key generation or saving fails
        """
        try:
            # Generate new Fernet key
            new_key = Fernet.generate_key()
            
            # Save to file
            key_file.write_bytes(new_key)
            
            # Set restrictive permissions (owner read/write only)
            key_file.chmod(0o600)
            
            logger.warning(
                f"🔐 NEW ENCRYPTION KEY GENERATED and saved to {key_file}\n"
                f"   ⚠️  CRITICAL: Back up this file! If lost, all encrypted credentials become unrecoverable.\n"
                f"   📋 Backup command: cp {key_file} {key_file}.backup\n"
                f"   🔒 Key has been set to read/write by owner only (600 permissions)"
            )
            
            return new_key
            
        except Exception as e:
            logger.error(f"Failed to generate and save encryption key: {e}")
            raise EncryptionKeyError(f"Could not generate encryption key: {e}")
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.
        
        Args:
            plaintext: String to encrypt
            
        Returns:
            str: Base64-encoded encrypted string suitable for database storage
            
        Raises:
            EncryptionKeyError: If encryption fails
        """
        if not plaintext:
            return ""
        
        try:
            encrypted_bytes = self.cipher.encrypt(plaintext.encode('utf-8'))
            return encrypted_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionKeyError(f"Failed to encrypt data: {e}")
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.
        
        Args:
            ciphertext: Base64-encoded encrypted string from database
            
        Returns:
            str: Decrypted plaintext string
            
        Raises:
            EncryptionKeyError: If decryption fails (invalid key or corrupted data)
        """
        if not ciphertext:
            return ""
        
        try:
            decrypted_bytes = self.cipher.decrypt(ciphertext.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except InvalidToken:
            logger.error("Decryption failed: Invalid token (wrong key or corrupted data)")
            raise EncryptionKeyError(
                "Could not decrypt credential. The encryption key may have changed or data is corrupted."
            )
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionKeyError(f"Failed to decrypt data: {e}")
    
    def encrypt_dict(self, data: dict) -> str:
        """
        Encrypt a dictionary by converting to JSON and encrypting.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            str: Encrypted JSON string
        """
        import json
        json_str = json.dumps(data, sort_keys=True)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, ciphertext: str) -> dict:
        """
        Decrypt and parse JSON dictionary.
        
        Args:
            ciphertext: Encrypted JSON string
            
        Returns:
            dict: Decrypted dictionary
        """
        import json
        plaintext = self.decrypt(ciphertext)
        return json.loads(plaintext)
    
    def rotate_key(self, new_key: bytes) -> None:
        """
        Rotate encryption key (for future implementation).
        
        Note: This requires re-encrypting all existing credentials.
        Not implemented in MVP - will be part of future PRD.
        
        Args:
            new_key: New Fernet key to use
        """
        raise NotImplementedError("Key rotation not implemented in MVP")
    
    def get_key_info(self) -> dict:
        """
        Get information about the current encryption key (for diagnostics).
        
        Returns:
            dict: Key information (does NOT include actual key)
        """
        key_file = Path(__file__).parent.parent / ".credential_key"
        
        return {
            "source": "environment_variable" if os.getenv("CREDENTIAL_ENCRYPTION_KEY") else "file",
            "key_file_exists": key_file.exists(),
            "key_file_path": str(key_file) if key_file.exists() else None,
            "key_file_permissions": oct(key_file.stat().st_mode)[-3:] if key_file.exists() else None,
            "algorithm": "Fernet (AES-128-CBC + HMAC-SHA256)"
        }


# Singleton instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """
    Get or create singleton encryption service instance.
    
    Returns:
        EncryptionService: Encryption service instance
    """
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service

