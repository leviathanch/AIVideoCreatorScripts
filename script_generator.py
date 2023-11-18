from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

video_gen_prompt = """
Make a 20 minute long video about how virtual AI girlfriends like Replika have automated gold diggers away,
because they also can just look cute and ask for money, in that case subscription fees, but without
the risk of divorce. Talk about what the future impacts might be, as soon as robots also can do the chores.
It should have at least 10 scenes.
"""

prefix = """You are an expert at creating structures for YouTube video scripts.
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
    tools =tools,
    verbose = True
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    verbose = True,
    memory = memory
)

result = agent_chain.run({'input':video_gen_prompt,'json_format':JSON_FORMAT})
with open("script_structure.json",'w') as f:
	f.write(result)
	f.close()
