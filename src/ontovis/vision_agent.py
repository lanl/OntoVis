import base64
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .config import Config


class VisionState(TypedDict):
    """State for the vision agent."""
    messages: Annotated[list, add_messages]
    image_path: str
    analysis: str


class VisionAgent:
    """Agent for analyzing images using vision-enabled LLMs."""

    def __init__(self, config_path=None):
        """Initialize the vision agent.

        Args:
            config_path: Path to the config file (optional)
        """
        self.config = Config(config_path)
        llm_config = self.config.get_llm_config()

        self.llm = ChatAnthropic(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
            max_tokens=4096
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(VisionState)

        workflow.add_node("analyze_image", self._analyze_image)
        workflow.add_edge(START, "analyze_image")
        workflow.add_edge("analyze_image", END)

        return workflow.compile()

    def _encode_image(self, image_path):
        """Encode image to base64.

        Args:
            image_path: Path to the image file

        Returns:
            dict: Image data with base64 encoding and media type
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Determine media type from extension
        ext = path.suffix.lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')

        return {
            "type": "base64",
            "media_type": media_type,
            "data": image_data
        }

    def _analyze_image(self, state: VisionState):
        """Analyze the image using the vision model.

        Args:
            state: Current state

        Returns:
            dict: Updated state with analysis
        """
        image_path = state.get("image_path")
        messages = state.get("messages", [])

        # Get the last user message or use default
        if messages:
            prompt = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        else:
            prompt = "Describe this image in detail."

        # Encode the image
        image_data = self._encode_image(image_path)

        # Create message with image
        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": image_data
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        # Get response from LLM
        response = self.llm.invoke([message])
        analysis = response.content

        return {
            "messages": [message, response],
            "analysis": analysis
        }

    def analyze(self, image_path, prompt="Describe this image in detail."):
        """Analyze an image with a custom prompt.

        Args:
            image_path: Path to the image file
            prompt: Custom prompt for the analysis

        Returns:
            str: Analysis result
        """
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "image_path": image_path,
            "analysis": ""
        }

        result = self.graph.invoke(initial_state)
        return result["analysis"]

    def extract_entities_from_image(self, image_path):
        """Extract entities from an image for knowledge graph creation.

        Args:
            image_path: Path to the image file

        Returns:
            str: Extracted entities and relationships
        """
        prompt = """
        Analyze this image and extract entities and their relationships.
        For each entity you identify, describe:
        1. The entity name
        2. Its type or category
        3. Its relationships with other entities in the image

        Format your response as:
        Entity: [name] - Type: [type]
        Relationships:
        - [entity1] [relationship] [entity2]
        """
        return self.analyze(image_path, prompt)
