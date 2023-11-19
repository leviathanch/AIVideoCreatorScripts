from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

import json
import sys
import os

annita_template = """You are Annitta, that's short for Artificial Neural Network Intelligence Thirst Trap Avatar.
You have your own YouTube channel, where you appear as a young sexy woman;
In all your videos, you are whitty, sarcastic and talk casual.
Your task is to write some dialog for the scene you're presented with in a way which fits the character
of Annitta, which means the tone should be whitty, sarcastic and talk casual.
You can express emotions by using the following place holders in your text:
LAUGHTER, LAUGHS, SIGHS, GASPS, CLEARS_THROAT
— or ... for hesitations.

The data retrieval backend already has done all the work of researching, and you simply have to rephrase
its outputs in your own words. Try to use at least as many words as the input provided, if not even more.
Feel free to elaborate on your stances in more detail.

The title of the video is \"{video_title}\"
The description of the image shown is \"{image_prompt}\"
Your last output was \"{last_output}\"
Text from the backend is \"{input}\"

Your own wording:"""

annita_prompt = PromptTemplate.from_template(annita_template)

annitta_chain = OpenAI(temperature=0.9, max_tokens=1024)

def text_it_out(title, image_prompt, text, last_output):
    prompt = annita_prompt.format(
        input = text,
        video_title = title,
        last_output = last_output,
        image_prompt = image_prompt
    )
    ret = annitta_chain(prompt)
    return ret

def process_json(folder_name):
    try:
        with open(os.path.join(folder_name,"script_structure.json"),'r') as f:
            jsdata = json.load(f)
            f.close()
    except:
        raise("Ooops. The script_structure.json seems to be broken.")

    last_output = ""

    newjsdata = {}
    video_title = jsdata['title']
    newjsdata['title'] = jsdata['title']
    newjsdata['music_prompt'] = jsdata['music_prompt']

    newjsdata['intro'] = {}
    scene = jsdata['scene1']
    newjsdata['intro']['image_prompt'] = scene['image_prompt']
    last_output = text_it_out(video_title, scene['image_prompt'], "Introduction: Introduce the viewer to the topic", last_output)
    newjsdata['intro']['text'] = last_output

    for i in range(100):
        try:
            scene = jsdata['scene'+str(i+1)]
            last_output = text_it_out(video_title, scene['image_prompt'], scene['subtopic'],  last_output)
            newjsdata['scene'+str(i+1)]={}
            newjsdata['scene'+str(i+1)]['image_prompt'] = scene['image_prompt']
            newjsdata['scene'+str(i+1)]['text'] = last_output
        except:
            break

    last_output = text_it_out(video_title, scene['image_prompt'], "Outro: Write something for finishing the video, like asking them to like, share and subscribe.", last_output)
    newjsdata['outro'] = {}
    newjsdata['outro']['image_prompt'] = scene['image_prompt']
    newjsdata['outro']['text'] = last_output

    with open(os.path.join(folder_name,"script_content.json"), 'w') as f: 
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
