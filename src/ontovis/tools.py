import base64
import code
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import requests
from typing import Optional, List
from typing_extensions import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI


logger = logging.getLogger("global_logger.tools")


class CodeExecutionTool:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.output_dir = Path("./generated_artifacts")
        self.output_dir.mkdir(exist_ok=True)
    
    def get_artifact_paths(self, artifacts: dict) -> List[str]:
        """Return paths to artifacts (already saved via volume mount)."""
        return [str(self.output_dir / filename) for filename in artifacts.keys()]


@tool
def execute_python_code(code: str, 
                        requirements: Optional[List[str]] = None, 
                        timeout: int = 30) -> str:
    """
    Execute Python code in a secure Docker container and retrieve generated artifacts.
    
    [... same docstring ...]
    """
    
    tool_instance = CodeExecutionTool()
    
    try:
        response = requests.post(
            f"{tool_instance.api_url}/execute",
            json={
                "code": code,
                "language": "python",
                "timeout": timeout,
                "requirements": requirements or []
            },
            timeout=timeout + 5
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Get file paths (files already exist via volume mount!)
        saved_files = []
        if result["artifacts"]:
            saved_files = tool_instance.get_artifact_paths(result["artifacts"])
        
        # Format response (same as before)
        output_parts = []
        
        if result["status"] == "success":
            output_parts.append("✅ Code executed successfully!")
        else:
            output_parts.append("❌ Code execution failed!")
        
        if result["stdout"]:
            output_parts.append(f"\n📝 Output:\n{result['stdout']}")
        
        if result["stderr"]:
            output_parts.append(f"\n⚠️ Errors/Warnings:\n{result['stderr']}")
        
        if saved_files:
            output_parts.append(f"\n📁 Generated files:")
            for file_path in saved_files:
                output_parts.append(f"  - {file_path}")
        
        output_parts.append(f"\n⏱️ Execution time: {result['execution_time']:.2f}s")
        
        return "\n".join(output_parts)
        
    except requests.exceptions.Timeout:
        return f"❌ Code execution timed out after {timeout} seconds"
    except requests.exceptions.RequestException as e:
        return f"❌ Error communicating with execution server: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"



@tool 
def image_analysis_tool(image_url: str, 
                        human_msg: str = "Provide a detailed explanation of this image", 
                        sys_msg: str = "You are a precise scientific image analysis assistant") -> str:
    """
    Analyze an image from a given URL and return what it contains.

    Args:
        image_url (str): URL of the image to analyze
        human_msg (str): Instructions or questions about the image
        sys_msg (str): System message defining the assistant's role

    Returns:
        str: Analysis result
    """
    print("\n-----image_analysis_tool---")
    print(f"\tImage URL: {image_url}, \n\thuman_msg: {human_msg}, \n\tsys_msg: {sys_msg}\n")
    logger.info(f"\n\n!!!image_analysis_tool :: analyzing image: {image_url}, \n\thuman_msg: {human_msg}, \n\tsys_msg: {sys_msg}\n\n")


    # Use a vision model to understand the image
    llm = ChatOpenAI(model="gpt-4o-mini")

    try:
        with open(image_url, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{image_b64}"
    except Exception as e:
        return f"Error loading image: {str(e)}"

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": sys_msg
            },
            
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    },
                    {
                        "type": "text",
                        "text": human_msg
                    }
                ]
            }
        ]
    )

    logger.info(f"Image analysis result: {response.content}")
    return response.content
    



@tool
def web_search_tool(query: str, 
                    max_results: int = 5) -> list:
    """
    Perform a web search on the topics and returns structured search results with title, snippet, and URL.

    Args:
        query (str): Search query
        max_results (int): Max number of results to return

    Returns:
        list of {title, snippet, url}
    """
    print("\n-----web_search_tool---\n")
    logger.info(f"\n\n!!!web_search_tool :: executing search for query: {query}\n\n")

    search_url = "https://duckduckgo.com/html/"
    params = {"q": query}

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ddg_web_search_tool/1.0)"
    }

    try:
        r = requests.get(search_url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return [{"error": f"Request failed: {str(e)}"}]

    soup = BeautifulSoup(r.text, "html.parser")
    results = []

    for a in soup.select(".result__a"):
        if len(results) >= max_results:
            break

        title = a.get_text()
        url = a.get("href")

        snippet_elem = a.find_parent("div", class_="result").select_one(".result__snippet")
        snippet = snippet_elem.get_text() if snippet_elem else ""

        results.append({
            "title": title,
            "snippet": snippet,
            "url": url,
            "source": "duckduckgo"
        })

    if not results:
        return [{"message": "No results found"}]

    return results



@tool
def python_repl_tool(code: Annotated[str, "Python code to execute. All generated outputs (plots, data, files) must be written to disk."]) -> str:
    """Executes arbitrary Python code. 
    The executed code must save any plots or files to disk, not return them directly.
    Returns the stdout/stderr logs as a string.

    Arg:
        code (str): the code to run

    Returns:
        str: error message
    """
    print("\n-----python_repl_tool---\n")
    logger.info(f"\n\n!!!python_repl_tool :: executing code: {code}\n\n")


    python_repl = PythonREPL()
    try:
        result = python_repl.run(code)  # returns captured stdout; plots display only if backend and show() cooperate
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    

    return f"Successfully executed:\n```python\n{code}\n```\nStdout:\n{result}"




@tool
def volume_rendering_instructions() -> str:
    """Provides instructions on how to properly perform volume rendering and save files for the VisAgent to access.
    
    Returns:
        str: instructions for volume rendering and file saving
    """
    print("\n-----volume_rendering_instructions---\n")
    logger.info(f"\n\n!!!volume_rendering_instructions :: executing instructions\n\n")

    instructions = """
        The Steps for volume rendering are:
        1. Load your 3D data (e.g., medical scan, scientific simulation output).
        2. Compute a histogram of the data to understand the value distribution.
        3. Use a grayscale colormap to visualize the histogram and identify value ranges of interest.
        4. Choose an appropriate transfer function based on the histogram to map data values to colors and opacities.
        5. Use a volume rendering library (e.g., VTK, Mayavi, PyVista) to render the volume with the chosen transfer function.
        6. Save the rendered image to disk in a location accessible to the VisAgent 
           
    """

    return instructions