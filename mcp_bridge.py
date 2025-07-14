from fastapi import FastAPI, Request
import subprocess

app = FastAPI()

@app.post("/execute")
async def execute_command(request: Request):
    data = await request.json()
    cmd = data['command']
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except Exception as e:
        return {"error": str(e)}