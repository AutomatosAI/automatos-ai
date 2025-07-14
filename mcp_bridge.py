from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
import subprocess
import logging
from json.decoder import JSONDecodeError
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

ALLOWED_COMMANDS = ["ls", "git", "npm", "docker", "pytest", "ssh", "echo", "cat", "python"]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
API_KEY = os.getenv("API_KEY")

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

@app.post("/run-task")
async def run_task(request: Request, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    try:
        data = await request.json()
        task = data.get('task', '')
        github_url = data.get('github_url', '')
        if not task or not github_url:
            raise ValueError("Task and GitHub URL required")
        orchestrator = Orchestrator()
        input_task = {"task": task, "github_url": github_url}
        result = orchestrator.run_flow(input_task)
        return {"result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Run task error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")