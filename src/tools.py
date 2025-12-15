from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import base64
from typing_extensions import Annotated

from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool


@tool 
def image_analysis_tool(image_url: str, human_msg: str = "Provide a detailed explanation of this image", sys_msg: str = "You are a precise scientific image analysis assistant") -> str:
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

    return response.content
    



@tool
def web_search_tool(query: str, max_results: int = 5) -> list:
    """
    Perform a web search on the topics and returns structured search results with title, snippet, and URL.

    Args:
        query (str): Search query
        max_results (int): Max number of results to return

    Returns:
        list of {title, snippet, url}
    """
    print("\n-----web_search_tool---\n")

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


    python_repl = PythonREPL()
    try:
        result = python_repl.run(code)  # returns captured stdout; plots display only if backend and show() cooperate
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout:\n{result}"




my_tools = [
    web_search_tool,
    image_analysis_tool,
    python_repl_tool
]