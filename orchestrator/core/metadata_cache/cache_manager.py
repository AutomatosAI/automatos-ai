"""
Cache Manager - Simple stub to keep existing code working
This will be replaced during proper migration.
"""

class CacheManager:
    """Stub cache manager"""
    def __init__(self, db, redis_client):
        self.db = db
        self.redis = redis_client
    
    def get_apps(self, category=None):
        """Stub - returns empty list"""
        return []
    
    def get_app_actions(self, app_name):
        """Stub - returns empty list"""
        return []
    
    def get_stats(self):
        """Stub - returns empty dict"""
        return {}
