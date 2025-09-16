import psutil
import time
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/system", tags=["system-health"])

@router.get("/metrics")
async def get_system_metrics():
    """Get real system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        return {
            "cpu": {"usage_percent": round(cpu_percent, 1)},
            "memory": {"usage_percent": round(memory.percent, 1)},
            "disk": {"usage_percent": round((disk.used / disk.total) * 100, 1)},
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
