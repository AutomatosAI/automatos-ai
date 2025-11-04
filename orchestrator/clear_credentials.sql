-- Clear encrypted credentials that can't be decrypted
-- This is safe for dev/MVP - you'll re-add credentials via the UI

BEGIN;

-- Show what will be deleted
SELECT id, name, credential_type_id, created_at 
FROM credentials 
WHERE encrypted_data IS NOT NULL;

-- Delete all encrypted credentials
DELETE FROM credentials WHERE encrypted_data IS NOT NULL;

-- Verify deletion
SELECT COUNT(*) as remaining_credentials FROM credentials;

COMMIT;

-- After running this, restart your backend and add credentials via Settings UI

