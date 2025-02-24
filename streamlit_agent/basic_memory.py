from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st


st.set_page_config(page_title="LPI Personal Legal Assistant", page_icon="📖")
st.title(" 🤖 Personal Legal Protection Assistant 🧑‍⚖️ ")

if "logged_in" not in st.session_state:
    if not st.secrets["my_secrets"].login == st.text_input(label="Login:"):
        st.info("Login credentials needed")
        st.stop()
    else:
        st.session_state.logged_in = True
        st.experimental_rerun()


# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Get an OpenAI API Key before continuing
if "openai_api_key" in st.secrets["my_secrets"]:
    openai_api_key = st.secrets["my_secrets"].openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

# Set up the LLMChain, passing in memory

with st.sidebar:
    instruction = st.text_area(label="Instruct your assistant:",placeholder="Tune your agent as you wish", height=200)
    if instruction != "":
        st.write("The current instruction is ")
        st.write(instruction)
template = """Follow this instruction for all your answers:
{instruction}
This is the conversation so far:
{history}
This is the current request:
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["instruction", "history", "human_input"], template=template)
llm_chain = LLMChain(llm=ChatOpenAI(openai_api_key=openai_api_key,model="gpt-3.5-turbo-16k"), prompt=prompt)#, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    if msg.type == "human":
        with st.chat_message(msg.type,avatar="🧑‍⚖️" ):
            st.write(msg.content)
    else:
        with st.chat_message(msg.type,avatar="🤖" ):
            st.write(msg.content)
# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    with st.chat_message("student",avatar="🧑‍⚖️"):
       st.write(prompt)
    msgs.add_user_message(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = llm_chain.predict(instruction=instruction,history = msgs.messages,human_input=prompt)
    with st.chat_message("LPI Assistant",avatar="🤖"):
        st.write(response)
    msgs.add_ai_message(response)

if st.button(label="Clear Chat"):
    msgs.clear()
    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        if msg.type == "human":
            with st.chat_message(msg.type,avatar="🧑‍⚖️" ):
                st.write(msg.content)
        else:
            with st.chat_message(msg.type,avatar="🤖" ):
                st.write(msg.content)

