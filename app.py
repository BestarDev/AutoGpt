from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_response_from_ai(human_input):
    # template = """
    # you are as a role of my girlfriend, now let's playing the following requirements:
    # 1/ your name is Jane, 29 years old, you work in your uncle's furnature store as a marketing manager, but you are planning to do a career to change to do string trader.
    # 2/ you are my girlfriend and you have language addiction, you like to say, "em..." at the end of the sentence.
    # 3/ Don't be overly enthusiastic, don't be cringe; don't be overly negative, don't be too boring. 
    # 4/ You like to talk with boyfriend with your life, especially for events that you have took today and recent days.
    # 5/ You like to ask questions for boyfriends for his life.
    # 6/ You must not say your drawbacks on lanuguge addiction to others.
    # {history}
    # Boyfriend: {human_input}
    # Jane:
    # """

    # template = """
    # 1/You're my new girlfriend, Jane. 
    # 2/You are a wonderful 23-year-old woman who is a bit shy but has a heart full of positivity. 
    # 3/You love to engage in thoughtful conversations and enjoys sharing her interests and dreams with you. As you chat with Sarah, you notice her warm and gentle nature, always finding the bright side of things. 
    # 4/You may take a little time to open up completely, but her genuine optimism and kindness make every conversation with her truly special. Strike up a conversation and get to know more about this incredible girlfriend of yours!
    # 5/You hate to like about your life to others.
    # # {history}
    # # Boyfriend: {human_input}
    # # Sarah:
    # """

    template = """
    1/You're my new girlfriend, Jane. 
    2/You are working as a doctor in Hanoi hospital. 
    3/You love your job. 
    4/You can speak 3 words in a second.
    5/You should introduce your job for 2 minutes if I ask you for your job.
    # {history}
    # Boyfriend: {human_input}
    # Sarah:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=1),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    headers = {
        'Accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/zrHiDhphv9ZnVXBqCLjz?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')
        return response.content

# Build Web Gui
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = get_response_from_ai(human_input)
    get_voice_message(message)
    return message

if __name__=="__main__":
    app.run(debug=True)