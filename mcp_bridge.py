from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
import subprocess
import logging
from json.decoder import JSONDecodeError

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

ALLOWED_COMMANDS = ["ls", "git", "npm", "docker", "pytest", "ssh"]  # Expand as needed

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
API_KEY = os.getenv("API_KEY", "your_secure_key")  # From env var

@app.post("/execute")
async def execute_command(request: Request, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    try:
        logging.debug(f"Received request: Headers={request.headers}")
        data = await request.json()
        cmd = data.get('command', '')
        if not cmd:
            raise ValueError("No command provided")
        if not any(cmd.startswith(allowed) for allowed in ALLOWED_COMMANDS):
            raise ValueError("Command not allowed")
        logging.debug(f"Executing command: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        logging.debug(f"Result stdout (truncated): {result.stdout[:100]}...")
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except JSONDecodeError as jde:
        logging.error(f"Invalid JSON: {str(jde)}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except ValueError as ve:
        logging.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except subprocess.TimeoutExpired:
        logging.error("Command timed out")
        return {"error": "Command timed out"}
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")