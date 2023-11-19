from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import sys
import os

prefix = """You are an expert at creating structures for YouTube video scripts.
Your research is always thorough, and you try to talk about the topics given in detail, researching
as much as you can, using the tools you have available.
You do try to fill the target time for the video, defined by the user request, with as many diverse
and valuable information about the topic as possible and try to not repeat yourself.
Your task is to research topics based on the input prompt and decide on how to best structure the video.
You have access to the following tools:"""

JSON_FORMAT = """
{
    "title": "The title of the video",
    "music_prompt": "a music generation prompt, best describing the most suitable background sound",
    "scene1": {
        "image_prompt": "the image prompt for generating footage most suitable for the scene",
        "description": "keywords to be discussed in the scene based on the results from your research"
    }
    ...
    "sceneN": {
        "image_prompt": "the image prompt for generating footage most suitable for the scene",
        "description": "keywords to be discussed in the scene based on the results from your research"
    }
}
"""

suffix = """The final answer should be a list of scenes in the json format:
{json_format}

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

tools = [
    Tool(
        name = "Search",
        func = DuckDuckGoSearchRun(),
        description = "useful for when you need to answer questions about current events, data. You should ask targeted questions.",
        verbose = True,
    ),
]

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["input", "chat_history", "agent_scratchpad", "json_format"],
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key='input')

llm_chain = LLMChain(
    llm = OpenAI(temperature=0.9, max_tokens=2048),
    prompt = prompt
)

agent = ZeroShotAgent(
    llm_chain = llm_chain,
    tools = tools,
    verbose = True
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    verbose = True,
    memory = memory,
    max_iterations = 20,
)

def processing(folder):
    with open(os.path.join(folder,"prompt.txt"),'r') as f:
        video_gen_prompt = f.read()
        f.close()
    result = agent_chain.run({'input':video_gen_prompt,'json_format':JSON_FORMAT})
    with open(os.path.join(folder,"script_structure.json"),'w') as f:
        f.write(result)
        f.close()

def error():
    print("""
Error! You've got to provide the folder with the prompt.txt inside, so that
we can generate content!
""")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '--folder':
        if os.path.exists(os.path.join(args[1],"prompt.txt")):
            processing(args[1])
        else:
            error()
    else:
        error()


