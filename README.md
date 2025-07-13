def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you?"
    elif "your name" in user_input:
        return "I'm ChatBot, your virtual assistant."
    elif "how are you" in user_input:
        return "I'm just code, but I'm running well!"
    elif "bye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm not sure how to respond to that."

# Chat loop
print("🤖 ChatBot is online! Type 'exit' to quit.")
while True:
    inp = input("You: ")
    if inp.lower() == "exit":
        print("ChatBot: Bye!")
        break
    response = chatbot_response(inp)
    print("ChatBot:", response)
    from transformers import pipeline
import gradio as gr

# Load a pre-trained conversational model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Define a function that handles the conversation
def chat_with_bot(message):
    response = chatbot(message, max_length=100, do_sample=True)[0]['generated_text']
    return response.strip()

# Create the Gradio web interface
gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(lines=2, placeholder="Ask something..."),
    outputs="text",
    title="AI ChatBot",
    description="Chat with an AI powered by DialoGPT"
).launch()
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai

load_dotenv()

st.set_page_config(
    page_title="chat with GENERATIVE AI",
    page_icon='🧠',
    layout="centered",
)

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

gen_ai.configure(api_key=GOOGLE_API_KEY)
model=gen_ai.GenerativeModel("gemini-2.0-flash")
def map_role(role):
    return "assistant" if role == "model" else role

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
    
st.title("Generative AI Chatbot")


for message in st.session_state.chat_session.history:
    with st.chat_message(map_role(message.role)):
        st.markdown(message.parts[0].text)

user_input= st.chat_input("Type your message here....")

if user_input:
    st.chat_message("user").markdown(user_input)
    response=st.session_state.chat_session.send_message(user_input)

    with st.chat_message("assistant"):
        st.markdown(response.text)
