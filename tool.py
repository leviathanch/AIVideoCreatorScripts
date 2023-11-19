from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

def tools():
    # Tools
    tools = [
        Tool(
            name = "Search",
            func = DuckDuckGoSearchRun(),
            description = "useful for when you need to answer questions about current events, data. You should ask targeted questions.",
            verbose = True,
        ),
    ]
    return tools
