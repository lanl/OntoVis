
import time
import yaml
from pathlib import Path
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict, Annotated, List
from IPython.display import Markdown, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, message_to_dict, messages_from_dict
from langchain_core.tools import tool

from ontovis.tools import (
    image_analysis_tool,
    web_search_tool,
    python_repl_tool,
    volume_rendering_instructions
)



##############################################################
#### Config Files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print("Loading config from:", PROJECT_ROOT / "configs/config.yaml")


with open(PROJECT_ROOT / "configs/config.yaml", "r") as f:
    config_data = yaml.safe_load(f)


##############################################################
#### Langraph agent nodes

class AgentState(TypedDict):
    """State for the DSIExplorer agent graph."""
    messages: Annotated[list, add_messages]
    response: str
    metadata: Dict[str, Any]



class VisWorker:
    """A visual data exploration agent using LangGraph."""

    def __init__(self, llm: ChatOpenAI, base_prompt: str, tools: List[ToolNode], name: str = "VisAgent"):
        self.llm = llm
        self.system_prompt = SystemMessage(content=base_prompt)
        self.llm = llm.bind_tools(tools)
        self.name = name  # add a name for easier tracing


    def __call__(self, state):
        messages = state["messages"]

        if not messages or not isinstance(messages[0], SystemMessage):  # add system prompt if it exists
            conversation = [self.system_prompt] + messages
        else:
            conversation = messages

        response = self.llm.invoke(conversation)
        
        return {
            "messages": messages + [response],
            "response": response.content,
            "metadata": response.response_metadata
        }


def should_call_tools(state: AgentState) -> str:
    """Decide whether to call tools or continue.

    Arg:
        state (AgentState): the current state of the graph

    Returns:
        str: "call_tools" or "continue"
    """

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "call_tools"
    
    return "continue"



class VisExplorer:
    def __init__(self, llm):
        self.llm = llm
        self.build_graph()


    def build_graph(self):
        vis_agent_prompt = config_data['vis_agent']['llm-prompt']

        my_tools = [
            web_search_tool,
            image_analysis_tool,
            python_repl_tool,
            volume_rendering_instructions,
        ]

        vis_agent = VisWorker(
            llm=self.llm,
            base_prompt=vis_agent_prompt,
            tools=my_tools
        )   

        tool_node = ToolNode(my_tools)


        # Create the agent
        workflow = StateGraph(AgentState)

        workflow.add_node("VisWorker", vis_agent)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "VisWorker")
        workflow.add_conditional_edges(
            "VisWorker",
            should_call_tools,
            {
                "call_tools": "tools",
                "continue": END,
            },
        )

        workflow.add_edge("tools", "VisWorker")

        memory = MemorySaver()   
        self.graph = workflow.compile(checkpointer=memory)


    def chat(self, query, thread_id="user_1234"):
        """Chat with the vis agent.
        
        Args:
            thread_id (str): Unique identifier for the conversation thread.
            query (str): User's query.
        """

        start = time.time()

        result = self.graph.invoke(
            {
                "messages": [
                    HumanMessage(content=query)
                ]
            },
            config = {
                "configurable": {"thread_id": thread_id}
            },
        )

        # Get and display the cleaned output
        response_text = result["response"] 
        cleaned_output = response_text.strip()
        display(Markdown(cleaned_output))

        # Get statistics
        elapsed = time.time() - start
        total_tokens = result["metadata"].get("token_usage", {}).get("total_tokens", 0)

        print(f"\nQuery took: {elapsed:.2f} seconds, total tokens used: {total_tokens}\n ")  