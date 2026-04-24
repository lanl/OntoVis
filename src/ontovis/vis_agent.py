
import time
import yaml
import os
import logging
import requests

from datetime import datetime
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
    load_anuerism_guidelines,
    volume_rendering_instructions,
    medical_imaging_rating_guidelines
)


##############################################################
#### Logs

logger = logging.getLogger("global_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False    

##############################################################
#### Langraph agent nodes

class AgentState(TypedDict):
    """State for the DSIExplorer agent graph."""
    messages: Annotated[list, add_messages]
    plan: str  # The plan created by the planning agent
    next_agent: str  # Which agent should execute next
    response: str
    metadata: Dict[str, Any]



class PlanningAgent:
    """Agent responsible for creating a plan to accomplish visualization tasks."""
    
    def __init__(self, llm: ChatOpenAI, workspace_path: str):
        self.llm = llm
        self.workspace_path = workspace_path
        
        self.system_prompt = SystemMessage(content=f"""You are a Planning Agent specialized in creating detailed plans for data visualization tasks.

Your responsibilities:
1. Analyze the user's request and break it down into clear, actionable steps
2. Identify what knowledge/guidelines are needed
3. Determine what type of visualization is required
4. Outline the analysis criteria for evaluating the results

When creating a plan:
- Be specific about data requirements
- Identify relevant guidelines or knowledge to retrieve
- Specify the visualization technique needed
- Define success criteria

Always structure your plan with clear numbered steps.
All generated files should be saved to: {workspace_path}
""")
    
    def __call__(self, state: AgentState):
        logger.info("=== PLANNING AGENT EXECUTING ===")
        
        # Get only the latest user message and analysis if iterating
        if state.get("analysis"):
            # We're iterating - create fresh context
            messages = [
                self.system_prompt,
                state["messages"][0] if state["messages"] else HumanMessage(content="Create a visualization"),
                HumanMessage(content=f"Previous analysis feedback:\n{state['analysis']}\n\nPlease update the plan accordingly.")
            ]
        else:
            # First time - use initial messages
            messages = [self.system_prompt] + [state["messages"][0]] if state["messages"] else [self.system_prompt]
        
        response = self.llm.invoke(messages)
        
        logger.info(f"Planning Agent created plan: {response.content[:200]}...")
        print(f"The plan is: {response.content}")
        
        return {
            "plan": response.content,
            # "next_agent": "knowledge_lookup",
            "messages": [response]
        }
    


class VisWorker:
    """A visual data exploration agent using LangGraph."""


    def __init__(self, 
                 llm: ChatOpenAI, 
                 base_prompt: str, 
                 tools: List[ToolNode], 
                 name: str = "VisAgent"):
        
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
    def __init__(self, 
                 llm, 
                 config_path: str = "",
                 workspace_prefix = "",
                 mode="jupyter"):
        self.llm = llm
        self.config_path = config_path
        self.mode = mode

        if self.config_path:
            try:
                with open(self.config_path, "r") as f:
                    self.config_data = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
                raise e


        self.workspace_path = self.create_workspace(workspace_prefix)

        log_file = os.path.join(self.workspace_path, "run.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter( "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        
        logger.info("!!!!!\n!!!!!Starting Vis agent initialization.")
        logger.info(f"Workspace created at: {self.workspace_path}")   
        logger.info(f"llm: {self.llm}")

        self.build_graph()



    def create_workspace(self, workspace_prefix="")-> str:
        """Create a workspace to store files and logs for the agent interaction
        
        Returns:
            str: the absolute path to the workspace
        """
        
        now = datetime.now()
        datetime_str = now.strftime("%Y_%m_%d__%H_%M")
        if workspace_prefix != "":
            workspace_prefix += "_"
        workspace_name = workspace_prefix + "vis_agent__" +  datetime_str
        absolute_workspace_path = os.path.abspath(workspace_name)
        print(f"Creating workspace at: {absolute_workspace_path}")

        # Create the woking directory and switch to it
        os.makedirs(absolute_workspace_path, exist_ok=True)
        os.chdir(absolute_workspace_path)  # changes the global working directory
        
        return absolute_workspace_path



    def build_graph(self):
        """Build the agent graph with the specified tools and prompts."""
        vis_agent_prompt = self.config_data['vis_agent']['llm-prompt']
        vis_agent_prompt += f"\n All files generated by the agent should be saved to the current working directory: {self.workspace_path}."

        logger.info(f"Building the agent graph with prompt: {vis_agent_prompt} and tools: {[tool.name for tool in [web_search_tool, image_analysis_tool, python_repl_tool, volume_rendering_instructions]]}")

        my_tools = [
            web_search_tool,
            image_analysis_tool,
            python_repl_tool,
            volume_rendering_instructions,
            medical_imaging_rating_guidelines,
        ]

        vis_agent = VisWorker(
            llm=self.llm,
            base_prompt=vis_agent_prompt,
            tools=my_tools
        )   

        tool_node = ToolNode(my_tools)


        # Create the agent
        workflow = StateGraph(AgentState)

        workflow.add_node("planning", planning_agent)
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
        logger.info(f"\n------------------------------\n!!!chat :: received query: {query}\n\n")

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

        logger.info(f"Agent response: {cleaned_output}")

        # Get statistics
        elapsed = time.time() - start
        total_tokens = result["metadata"].get("token_usage", {}).get("total_tokens", 0)

        print(f"\nQuery took: {elapsed:.2f} seconds, total tokens used: {total_tokens}\n ")  