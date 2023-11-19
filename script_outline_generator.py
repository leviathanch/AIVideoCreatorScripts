from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
)

from tool import tools

import sys
import os

prefix = """You are an expert at creating outlines for YouTube video scripts.
Your research is always thorough, and you try to talk about the topics given, in detail, researching
as much as you can, using the tools you have available.
You do try to fill the target time for the video, defined by the user request, with as many diverse
and valuable information about the topic as possible and try to not repeat yourself.
Your task is to research topics based on the input prompt and decide on how to best structure the video.
Your Final Answer always is the form
`Outline:

1. Introduction 
a)
...
2. Topic 1
a)
...
N. Conclusion
a) Recap & Summary 
...
`

You have access to the following tools:"""

suffix = """Begin!

Summary of conversation:
{history}
Current conversation:
{chat_history}
Question: {input}
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
    input_variables = ["input","history", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(
    llm = OpenAI(temperature=0.9, max_tokens=2048),
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
    max_iterations = 10,
    handle_parsing_errors = True,
)

def processing(folder):
    with open(os.path.join(folder,"prompt.txt"),'r') as f:
        video_gen_prompt = f.read()
        f.close()
    result = agent_chain.run(input=video_gen_prompt)
    with open(os.path.join(folder,"outline.txt"),'w') as f:
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


