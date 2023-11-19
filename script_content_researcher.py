from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
)

from tool import tools

import json
import sys
import os

prefix = """You are an expert in researching information on the internet, in order to
make sure, that everything in the YouTube videos we produce is factual and correct.
Your task is, to research information about the scene and title for the YouTube video
we're making, and provide long and detailed elaborations, based on your research results.

You have access to the following tools:"""

suffix = """Begin!

Summary of conversation:
{history}
Current conversation:
{chat_history}
The title of the video is \"{title}\"
The description of the scene is \"{input}\"
Thought:{agent_scratchpad}
"""

# Memory
conv_memory = ConversationBufferMemory(
    memory_key = "chat_history",
    input_key = 'input'
)

summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")
# Combined
memory = CombinedMemory(memories=[conv_memory, summary_memory])

# Agent
prompt = ZeroShotAgent.create_prompt(
    tools(),
    prefix = prefix,
    suffix = suffix,
    input_variables = ["input", "history", "chat_history", "agent_scratchpad", "title"],
)

llm_chain = LLMChain(
    llm = OpenAI(temperature=0.9, max_tokens=512),
    prompt = prompt
)

agent = ZeroShotAgent(
    llm_chain = llm_chain,
    tools = tools(),
    verbose = True
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools(),
    verbose = True,
    memory = memory,
    max_iterations=20,
)

def text_it_out(description, title):
    agent_text = agent_chain.run({
        'input':description,
        'title':title,
    })
    return agent_text

def process_json(folder_name):
    try:
        with open(os.path.join(folder_name,"script_structure.json"),'r') as f:
            jsdata = json.load(f)
            f.close()
    except:
        raise("Ooops. The script_structure.json seems to be broken.")

    newjsdata = {}
    video_title = jsdata['title']
    newjsdata['title'] = jsdata['title']
    newjsdata['music_prompt'] = jsdata['music_prompt']

    newjsdata['intro'] = {}
    scene = jsdata['scene1']
    newjsdata['intro']['image_prompt'] = scene['image_prompt']
    newjsdata['intro']['text'] = text_it_out("Introduction: Introduce the viewer to the topic", video_title)

    for i in range(100):
        try:
            scene = jsdata['scene'+str(i+1)]
        except:
            break

        newjsdata['scene'+str(i+1)]={}
        newjsdata['scene'+str(i+1)]['image_prompt'] = scene['image_prompt']
        newjsdata['scene'+str(i+1)]['text'] = text_it_out(scene['subtopic'], video_title)

    newjsdata['outro'] = {}
    newjsdata['outro']['image_prompt'] = scene['image_prompt']
    newjsdata['outro']['text'] = text_it_out("Outro: Write something for finishing the video, like asking them to like, share and subscribe.", video_title)

    with open(os.path.join(folder_name,"script_content_draft.json"), 'w') as f: 
        f.write(json.dumps(newjsdata, indent=4))
        f.close()

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '--folder':
        process_json(args[1])
    else:
        print("""
Error! You've got to provide the folder with the script_structure.json inside, so that
we can generate content!
""")
