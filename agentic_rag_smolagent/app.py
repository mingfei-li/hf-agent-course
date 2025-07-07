import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, OpenAIServerModel

from retriever import guest_info_tool
from tools import search_tool, weather_info_tool, hub_stats_tool

load_dotenv()
model = OpenAIServerModel(
    model_id="gpt-4o",  # or "gpt-3.5-turbo", "gpt-4", etc.
    api_key=os.getenv("OPENAI_API_KEY"),
)
alfred = CodeAgent(
    tools=[guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    model=model,
    add_base_tools=True,
    planning_interval=3
)

try:
    while True:
        user_query = input("What do you want to ask Alfred?\n")
        response = alfred.run(user_query, reset=False)

        print("Alfred's Response:")
        print(response)
except EOFError:
    print("\nExiting on EOF (Ctrl+D)")