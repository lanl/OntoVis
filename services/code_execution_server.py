import os
import uuid
import subprocess
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExecutionRequest(BaseModel):
    code: str
    requirements: List[str] = []
    timeout: int = 30

class ExecutionResponse(BaseModel):
    success: bool
    output: str
    error: str
    artifacts: List[str]

# Use shared volume instead of /tmp
WORKSPACE_BASE = os.environ.get("WORKSPACE_DIR", "/workspace")


@app.get("/download/{execution_id}/{filename}")
async def download_file(execution_id: str, filename: str):
    """Download a generated artifact."""
    artifacts_dir = "/app/generated_artifacts"
    file_path = os.path.join(artifacts_dir, f"{execution_id}_{filename}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.post("/execute", response_model=ExecutionResponse)
async def execute_code(request: ExecutionRequest):
    execution_id = str(uuid.uuid4())
    workspace_dir = os.path.join(WORKSPACE_BASE, execution_id)
    artifacts_dir = "/app/generated_artifacts"
    
    try:
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Write the code to a file
        script_path = os.path.join(workspace_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(request.code)
            f.flush()
            os.fsync(f.fileno())
        
        # Verify file was created
        if not os.path.exists(script_path):
            raise Exception(f"Failed to create script at {script_path}")
        
        # Build docker command - mount the shared volume
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--memory", "512m",
            "--cpus", "1",
            "-v", f"ontovis_execution-workspace:/workspace",
            "-w", f"/workspace/{execution_id}",
        ]
        
        # Only disable network if no requirements needed
        if not request.requirements:
            docker_cmd.extend(["--network", "none"])
        
        docker_cmd.extend([
            "python:3.11-slim",
            "sh", "-c"
        ])
        
        # Create command that installs requirements then runs script
        if request.requirements:
            packages = " ".join(request.requirements)
            run_cmd = f"pip install -q --no-cache-dir {packages} && python script.py"
        else:
            run_cmd = "python script.py"
        
        docker_cmd.append(run_cmd)
        
        # Execute
        start_time = time.time()
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=request.timeout
        )
        execution_time = time.time() - start_time
        
        # Collect artifacts
        artifacts = []
        for file in os.listdir(workspace_dir):
            if file not in ["script.py"]:
                src = os.path.join(workspace_dir, file)
                dst = os.path.join(artifacts_dir, f"{execution_id}_{file}")
                shutil.copy2(src, dst)
                artifacts.append(file)
        
        # Always include script.py in artifacts list
        artifacts.append("script.py")
        
        success = result.returncode == 0
        return ExecutionResponse(
            success=success,
            output=result.stdout,
            error=result.stderr,
            artifacts=artifacts
        )
        
    except subprocess.TimeoutExpired:
        return ExecutionResponse(
            success=False,
            output="",
            error="Execution timed out",
            artifacts=[]
        )
    except Exception as e:
        return ExecutionResponse(
            success=False,
            output="",
            error=str(e),
            artifacts=[]
        )
    finally:
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
