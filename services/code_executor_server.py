from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import docker
import tempfile
import shutil
import os
import base64
from typing import Optional, List

app = FastAPI()


class CodeExecutionRequest(BaseModel):
    code: str
    requirements: Optional[List[str]] = None
    timeout: Optional[int] = 60


class CodeExecutionResponse(BaseModel):
    output: str
    error: Optional[str] = None
    files: Optional[dict] = None  # {filename: base64_encoded_content}


@app.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """Execute Python code in a Docker container and return output + files."""
    
    client = docker.from_env()
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a temporary Python script
        script_path = os.path.join(temp_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(request.code)
        
        # Create requirements.txt if needed
        if request.requirements:
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, "w") as f:
                f.write("\n".join(request.requirements))
        
        # Create a Dockerfile for code execution
        dockerfile_content = """
FROM python:3.11-slim
WORKDIR /app
COPY script.py .
"""
        if request.requirements:
            dockerfile_content += """
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
"""
        dockerfile_content += """
RUN mkdir -p /app/output
CMD ["python", "script.py"]
"""
        
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        # Build the Docker image
        image, build_logs = client.images.build(
            path=temp_dir,
            tag=f"code-executor-{os.path.basename(temp_dir)}",
            rm=True
        )
        
        # Run the container
        container = client.containers.run(
            image.id,
            volumes={output_dir: {'bind': '/app/output', 'mode': 'rw'}},
            detach=True,
            remove=False
        )
        
        # Wait for container to finish
        result = container.wait(timeout=request.timeout)
        
        # Get logs
        stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
        stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
        
        # Collect generated files
        files = {}
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    files[filename] = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup
        container.remove(force=True)
        client.images.remove(image.id, force=True)
        
        return CodeExecutionResponse(
            output=stdout,
            error=stderr if stderr else None,
            files=files if files else None
        )
        
    except docker.errors.ContainerError as e:
        raise HTTPException(status_code=500, detail=f"Container error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
