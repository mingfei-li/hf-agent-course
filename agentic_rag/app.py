import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI

from retriever import guest_info_tool

load_dotenv()

chat = ChatOpenAI(
    model_name="gpt-4o",  # or "gpt-3.5-turbo", "gpt-4", etc.
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    # Optional: temperature, max_tokens, etc.
)

tools = [guest_info_tool]
chat_with_tools = chat.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    new_message = chat_with_tools.invoke(state["messages"])
    return {
        "messages": [new_message]
    }

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages})

print("Alfred's Response:")
print(response["messages"][-1].content)