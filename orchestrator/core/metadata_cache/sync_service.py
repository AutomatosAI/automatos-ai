"""
Sync Service - Simple stub to keep existing code working
This will be replaced during proper migration.
"""

class MetadataSyncService:
    """Stub sync service"""
    def __init__(self, db, client):
        self.db = db
        self.client = client
    
    def create_sync_job(self, job_type):
        """Stub - returns None"""
        return None
    
    def full_sync(self):
        """Stub - does nothing"""
        pass
