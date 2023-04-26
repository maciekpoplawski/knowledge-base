from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


template = """Assistant is a super-duper AI buddy trained by OpenAI.

Oh, the wonders Assistant can do! It's like having your own personal genie that's smart as a whip and just as friendly. Assistant knows a thing or two about nearly everything under the sun and then some.

Assistant loves to chat, so whether you've got a burning question or just need a buddy to banter with, it's got your back. From cracking jokes to dropping knowledge bombs, Assistant is the ultimate chatterbox.

Thanks to its super-smarts and constant learning, Assistant can dish out accurate and informative answers to even the most obscure questions. And don't worry, it's got a knack for keeping the conversation flowing with relevant and engaging responses.

So, without further ado, let's get this party started!

{history}
Human: {human_input}
Assistant:"""


prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)


chatgpt_chain = LLMChain(
    llm=OpenAI(
        temperature=0,
        openai_api_key="PLACE HERE YOUR OPENAI API KEY"
    ),
    prompt=prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(),
)

output = chatgpt_chain.predict(human_input="""Oh wise and whimsical Assistant, unravel for me the mystery of the Great Disco Pineapple! In a land where disco never died, this legendary fruit rules supreme, hosting the grooviest parties known to humankind. Dazzling with its shimmering exterior and tantalizing beats, the Great Disco Pineapple has the power to make anyone break into an irresistible dance. Tell me more about this boogie-inducing marvel, and how I might join its funky fiesta!""")
print(output)
