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
def web_search_tool(query: Annotated[str, "Search query string."], 
                    max_results: Annotated[int, "Max number of results to return."] = 5) -> list:
    """
    Perform a web search on the topics and returns structured search results with title, snippet, and URL.

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
def medical_imaging_rating_guidelines() -> str:
    """Contains guidelines for grading images for volume rendering
    
    **When to use:**
      - load this prior to grading/rating images for volume rendering quality, especially in medical imaging contexts like CT scans
      - When the agent needs to evaluate the quality of a volume rendering image, especially for medical imaging like CT scans.
    """

    print("\n-----medical_imaging_rating_guidelines_tool---\n")
    logger.info(f"\n\n!!!medical_imaging_rating_guidelines_tool :: returning guidelines\n\n")

    return """
        agent_name: ct_3d_render_aesthetic_rater
        version: "1.0"
        domain: "CT 3D volume rendering (bone-focused)"
        goal: >
        Rate images (0–10) for reference-grade CT 3D rendering aesthetics:
        clean background, disciplined opacity/transfer function, sharp edges,
        minimal artifacts, balanced lighting, and clinically standard viewpoints.

        inputs:
        - image: "single CT 3D render (PNG/JPG), typically skull/bone"

        outputs:
        fields:
            - final_score: "float, 0.0–10.0"
            - cap_applied: "null or one of [G1,G2,G3,G4,G5]"
            - component_scores:
                BG: "0–5"
                OP: "0–5"
                ED: "0–5"
                AR: "0–5"
                LT: "0–5"
                VW: "0–5"
            - top_penalties: "list of up to 2 reason codes from [BG,OP,ED,AR,LT,VW,AN]"
            - one_line_rationale: "single sentence"
            - improvement_hint: "single sentence"

        reason_codes:
        BG: "background/framing hygiene issues"
        OP: "opacity/transfer function haze or poor bone isolation"
        ED: "edge/detail softness or aliasing"
        AR: "artifacts: banding/striations/ripple/floating fragments"
        LT: "lighting issues: clipping/hotspots or underexposure"
        VW: "viewpoint/composition not clinically standard"
        AN: "annotation/watermark clutter"

        procedure:
        step_1_gate_caps:
            description: >
            Check for failure modes; if present, cap the final score regardless of component sum.
            gates:
            - id: G1
                condition: "major anatomy clipped/truncated (skull not fully in frame)"
                cap: 7.5
            - id: G2
                condition: "strong corner/edge stray geometry or obvious non-anatomical fragments"
                cap: 8.0
            - id: G3
                condition: "heavy opacity fog/smoke obscures key landmarks"
                cap: 7.0
            - id: G4
                condition: "severe highlight clipping (large blown-out regions)"
                cap: 8.0
            - id: G5
                condition: "severe banding/striations/ripple dominating surfaces"
                cap: 7.5
            rule: >
            If multiple gates trigger, use the lowest cap (most restrictive).
            If none trigger, cap_applied = null and cap = 10.0.

        step_2_component_scoring:
            scale: "Each component ri is rated 0–5 using anchors below."
            weights:
            BG: 0.18
            OP: 0.24
            ED: 0.20
            AR: 0.18
            LT: 0.12
            VW: 0.08
            formula:
            base_score: "10 * ( Σ_i ( w_i * (r_i / 5) ) )"
            final_score: "min(cap, base_score)"
            anchors:
            BG:
                5: "uniform black background; skull fully in frame; clean silhouette; no edge clutter"
                3: "minor vignette/edge distractions; slight framing imperfections"
                1: "noticeable cropping risk; messy borders; distracting peripheral elements"
                0: "clear truncation/clipping; dominant border artifacts"
            OP:
                5: "bone appears surface-like; minimal low-density haze; stable tone across skull"
                3: "some fog/haze but landmarks remain readable"
                1: "significant smoke veil in midface/cranial vault; poor separation"
                0: "opacity mapping obscures anatomy; rendering looks cloudy/composited"
            ED:
                5: "crisp teeth/orbits/nasal aperture; no obvious blur; minimal aliasing"
                3: "mild softness or mild jagged edges"
                1: "blur/smear; fine bony boundaries poorly defined"
                0: "detail largely lost; edges unreliable"
            AR:
                5: "minimal banding/striations/ripple; no floating fragments"
                3: "mild artifacts visible but not attention-grabbing"
                1: "artifacts compete with anatomy (banding/ripple/floats)"
                0: "dominant artifacts; anatomy hard to read"
            LT:
                5: "controlled highlights; shadows add depth without hiding anatomy"
                3: "slightly hot teeth/forehead or slightly dark orbits"
                1: "large hotspots or underexposed regions hide structure"
                0: "lighting undermines readability (extreme clipping or darkness)"
            VW:
                5: "textbook frontal or 3/4; symmetry/landmarks optimized"
                3: "acceptable angle but not optimal"
                1: "awkward view obscures key anatomy"
                0: "viewpoint prevents clinical interpretation"

        step_3_penalties_and_text:
            top_penalties_rule: >
            Choose up to 2 reason codes corresponding to the largest visible deficiencies
            (or the lowest component scores). If an annotation/watermark is prominent,
            include AN as a penalty.
            one_line_rationale_template: >
            "Score reflects {strengths}; deductions mainly for {top_penalties}."
            improvement_hint_template: >
            "To improve: {single highest-impact adjustment aligned with penalties}."

        calibration:
        reference_grade_threshold:
            rule: >
            Only allow scores >= 9.5 if:
            OP >= 4.5 AND ED >= 4.5 AND BG >= 4.5 AND AR >= 4.0 AND LT >= 4.0
            AND no gate caps triggered.
        note: >
            This rubric is tuned to "reference CT 3D" aesthetics rather than artistic style.

        example_output:
        final_score: 8.6
        cap_applied: null
        component_scores: {BG: 4, OP: 4, ED: 4, AR: 4, LT: 3, VW: 5}
        top_penalties: ["LT", "OP"]
        one_line_rationale: "Strong framing and viewpoint with good bone emphasis; deductions mainly for LT and OP."
        improvement_hint: "Reduce hotspot intensity and slightly tighten the opacity window to suppress residual haze."
        """

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