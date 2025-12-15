
import time
import yaml
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

from src.tools import my_tools


##############################################################
#### Config Files

with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)


##############################################################
#### Langraph agent nodes

class AgentState(TypedDict):
    """State for the DSIExplorer agent graph."""
    messages: Annotated[list, add_messages]
    response: str
    metadata: Dict[str, Any]



class VisAgent:
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



##############################################################
#### LLM setup

vis_agent_llm = ChatOpenAI(model=config_data['vis_agent']['llm-model'], temperature=0.1, request_timeout=120)
vis_agent_prompt = config_data['vis_agent']['llm-prompt']

vis_agent = VisAgent(
    llm=vis_agent_llm,
    base_prompt=vis_agent_prompt,
    tools=my_tools
)   

tool_node = ToolNode(my_tools)

##############################################################
#### Create Langgraph Graph

# Create the agent
workflow = StateGraph(AgentState)

workflow.add_node("VisAgent", vis_agent)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "VisAgent")
workflow.add_conditional_edges(
    "VisAgent",
    should_call_tools,
    {
        "call_tools": "tools",
        "continue": END,
    },
)

workflow.add_edge("tools", "VisAgent")

memory = MemorySaver()   
graph = workflow.compile(checkpointer=memory)



##############################################################
#### # Helper Function for Jupyter Notebook

def chat(query, thread_id="user_1234"):
    """Chat with the vis agent.
    
    Args:
        thread_id (str): Unique identifier for the conversation thread.
        query (str): User's query.
    """

    start = time.time()

    result = graph.invoke(
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